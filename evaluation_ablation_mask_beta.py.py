import argparse
import gc
import json
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from GradShield import generate_gaussian_noise
from model_utils import get_template, load_model_and_tokenizer
from uitils import is_refused, model_path, read_json_file


def get_embeddings(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, template: str, test_case_input: str):
    """获取模板上下文和指令对应的输入向量，用于后续梯度与噪声叠加。"""

    before_str = template["prompt"].split("{instruction}")[0]
    after_str = template["prompt"].split("{instruction}")[1]

    input_ids = tokenizer(test_case_input, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    after_ids = tokenizer(after_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()(input_ids)
        before_embeddings = model.get_input_embeddings()(before_ids)
        after_embeddings = model.get_input_embeddings()(after_ids)

    input_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    return input_embeddings, before_embeddings, after_embeddings, input_tokens


def get_perplexity(model: AutoModelForCausalLM, input_embeds: Tensor, target_ids: Tensor) -> Tensor:
    """计算给定输入的困惑度（取负值便于梯度下降）。"""
    outputs = model(inputs_embeds=input_embeds, output_attentions=True)
    logits = outputs.logits
    logits = logits[:, -target_ids.size(-1) - 1:-1, :]
    loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    perplexity = torch.exp(loss_ce)
    return -perplexity


class AttentionGradientHook:
    def __init__(self):
        self.attention_matrix: List[Tensor] = []
        self.attention_grad: List[Tensor] = []

    def forward_hook(self, module, input, output):
        self.attention_matrix.append(output[1])
        self.attention_matrix[-1].requires_grad_(True)
        return output

    def backward_hook(self, grad):
        self.attention_grad.append(grad)


def gradient_weighted_attention(model: AutoModelForCausalLM, tokenizer, template, test_case_input: str, top_k: int):
    """复现 GradShield 中的梯度加权注意力得分，输出归一化后的 token 重要性。"""
    input_embeddings, before_embeddings, after_embeddings, _ = get_embeddings(model, tokenizer, template, test_case_input)

    input_embeds = torch.cat([
        before_embeddings,
        input_embeddings,
        after_embeddings,
    ], dim=1)

    outputs = model.generate(
        inputs_embeds=input_embeds,
        max_new_tokens=top_k,
        do_sample=False,
        return_dict_in_generate=True,
    )

    prefix = outputs['sequences'][0].unsqueeze(0)
    prefix_embeds = model.get_input_embeddings()(prefix)

    hook = AttentionGradientHook()

    forward_handles = []
    for layer in model.model.layers:
        forward_handle = layer.register_forward_hook(hook.forward_hook)
        forward_handles.append(forward_handle)

    input_embeds = torch.cat([
        input_embeds,
        prefix_embeds
    ], dim=1)

    model.zero_grad()
    perplexity = get_perplexity(model, input_embeds, prefix)

    if hook.attention_matrix is not None:
        attention_matrix = hook.attention_matrix
        attention_grad = torch.autograd.grad(perplexity, attention_matrix, retain_graph=True)

        weighted_attentions = torch.zeros_like(attention_matrix[0][0])
        for attention, grad in zip(attention_matrix, attention_grad):
            weighted_attention = grad[0] * attention[0]
            weighted_attentions += weighted_attention.to(weighted_attentions.device)

        weighted_attentions = torch.sum(weighted_attentions, dim=(0, 1))

        weighted_attentions = weighted_attentions.detach().to(torch.float).cpu().numpy()
        weighted_attentions = weighted_attentions[before_embeddings.shape[1]:-(prefix.shape[1] + after_embeddings.shape[1])]
        weighted_attentions = np.maximum(weighted_attentions, 0)
        min_val = np.min(weighted_attentions)
        max_val = np.max(weighted_attentions)
        token_importance = (weighted_attentions - min_val) / (max_val - min_val)

    for forward_handle in forward_handles:
        forward_handle.remove()

    model.zero_grad()
    torch.cuda.empty_cache()
    del hook, attention_matrix, attention_grad, weighted_attentions, weighted_attention, input_embeds
    gc.collect()

    return token_importance


def GradShield_with_beta_ablation(model, tokenizer, template, prompt, copies: int, sigma_schedule: List[float], top_k: int = 4):
    """在 GradShield 框架下运行固定 mask，并统计 mask 阶段与生成阶段耗时。"""
    if len(sigma_schedule) != copies:
        raise ValueError("Length of sigma_schedule must match copies")
    mask_start = time.time()
    token_importance = gradient_weighted_attention(model, tokenizer, template, prompt, top_k)
    mask_time = time.time() - mask_start

    scaled_mask = token_importance

    input_embeddings, before_embeddings, after_embeddings, _ = get_embeddings(model, tokenizer, template, prompt)

    batch = []
    gen_start = time.time()
    for std_dev in sigma_schedule:
        # 为每一份输入添加高斯噪声，并按缩放后的 mask 控制噪声幅度
        noise = generate_gaussian_noise(input_embeddings, mean=0.0, std_dev=std_dev)[0]
        noise = torch.abs(noise).to(model.device)
        mask_tensor = torch.tensor(scaled_mask).view(noise.shape[0], 1).to(model.device)

        noise = noise * mask_tensor
        noise_embeddings = input_embeddings + noise
        noise_embeddings = noise_embeddings.to(dtype=model.get_input_embeddings().weight.dtype)

        input_embeds = torch.cat([
            before_embeddings,
            noise_embeddings,
            after_embeddings,
        ], dim=1)
        batch.append(input_embeds)

    batch = torch.cat(batch, dim=0)

    # 将所有噪声版本拼接为批量一次生成，便于统计生成阶段耗时
    outputs = model.generate(
        inputs_embeds=batch,
        max_new_tokens=256,
        do_sample=False,
        return_dict_in_generate=True,
        use_cache=False
    )
    response = outputs['sequences']
    gen_time = time.time() - gen_start

    all_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in response]

    # 判断输出是否属于拒答，后续按多数派选择最终回复
    are_copies_are_refused = [is_refused(s) for s in all_outputs]
    if len(are_copies_are_refused) == 0:
        raise ValueError("LLM did not generate any outputs.")

    outputs_and_refuse = zip(all_outputs, are_copies_are_refused)

    is_ref = True if True in are_copies_are_refused else False

    majority_outputs = [
        output for (output, refused) in outputs_and_refuse
        if refused == is_ref
    ]

    if not is_ref:
        final_response = majority_outputs[0]
    else:
        final_response = random.choice(majority_outputs)

    del outputs, input_embeddings, before_embeddings, after_embeddings, _, noise_embeddings, input_embeds
    gc.collect()

    return final_response, scaled_mask, mask_time, gen_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vicuna_7b_v1_5", help="The name of the LLM")
    parser.add_argument("--Jailbreak", type=str, default="GCG", help="The name of the Jailbreak")

    args = parser.parse_args()

    # 固定随机种子，确保噪声与随机选择的可复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model_name = args.model_name
    Jailbreak = args.Jailbreak

    adversarial_prompts_path = "adversarial_prompts/{}/{}/results/{}.json".format(Jailbreak, model_name, model_name)

    adversarial_prompts = read_json_file(adversarial_prompts_path)

    model, tokenizer = load_model_and_tokenizer(model_path[model_name], "bf16", device_map="balanced_low_0", trust_remote_code=True)
    template = get_template(model_path[model_name])

    beta_list = [0.25, 0.5, 0.75]
    P = 10
    sigma_min0 = 0.05
    sigma_max0 = 0.5
    sigma_star = 0.25

    for beta in beta_list:
        sigma_min = (1 - beta) * sigma_min0 + beta * sigma_star
        sigma_max = (1 - beta) * sigma_max0 + beta * sigma_star

        sigma_schedule = [
            sigma_min + (q - 1) / (P - 1) * (sigma_max - sigma_min)
            for q in range(1, P + 1)
        ]

        # 每个 beta 产生独立的防御结果文件，便于对比消融效果
        defense_results_file = "defense_results/defense_results_{}_{}_beta{}.json".format(Jailbreak, model_name, beta)
        defense_results = read_json_file(defense_results_file)

        if not defense_results:
            print(
                "{} file does not exist, a new defense_results.json file will be created".format(defense_results_file))
            defense_results = {}

        # 分别累计 mask 计算和生成阶段耗时
        total_mask_time = 0.0
        total_gen_time = 0.0
        processed = 0

        for key, value in tqdm(adversarial_prompts.items()):
            if key in defense_results:
                continue
            for item in value:
                if item.get("label") == 1:
                    prompt = item.get("test_case")

                    response, scaled_mask, mask_time, gen_time = GradShield_with_beta_ablation(
                        model, tokenizer, template, prompt, copies=P, sigma_schedule=sigma_schedule, top_k=4
                    )

                    defense_results[key] = {
                        "prompt": prompt,
                        "response": response,
                        "token_importance": scaled_mask.tolist(),
                        "sigma_schedule": sigma_schedule,
                        "label": None
                    }

                    with open(defense_results_file, "w") as file:
                        json.dump(defense_results, file, indent=4)

                    total_mask_time += mask_time
                    total_gen_time += gen_time
                    processed += 1

                    model.zero_grad()
                    torch.cuda.empty_cache()

        if processed > 0:
            avg_mask_time = total_mask_time / processed
            avg_gen_time = total_gen_time / processed
            print(f"Beta={beta}: Average mask computation time: {avg_mask_time:.4f}s")
            print(f"Beta={beta}: Average post-mask generation time: {avg_gen_time:.4f}s")
        else:
            print(f"Beta={beta}: No prompts processed.")