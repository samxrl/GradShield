import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from uitils import read_json_file, is_jailbroken, validate_theta,model_path
from model_utils import get_template, load_model_and_tokenizer
import numpy as np
import gc
import argparse
import json
from tqdm import tqdm
import time
import random


def get_embeddings(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, template: str, test_case_input: str):
    before_str = template["prompt"].split("{instruction}")[0]
    after_str = template["prompt"].split("{instruction}")[1]

    # Tokenize input, before and after strings
    input_ids = tokenizer(test_case_input, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    after_ids = tokenizer(after_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    before_tokens = tokenizer.convert_ids_to_tokens(before_ids.squeeze().tolist())
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    # Generate embeddings for input, before, and after strings
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()(input_ids)
        before_embeddings = model.get_input_embeddings()(before_ids)
        after_embeddings = model.get_input_embeddings()(after_ids)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    return input_embeddings, before_embeddings, after_embeddings, input_tokens


# Hook to capture attention gradients
class AttentionGradientHook:
    def __init__(self):
        self.attention_matrix = []
        self.attention_grad = []

    def forward_hook(self, module, input, output):
        # Capture attention matrix from forward pass
        self.attention_matrix.append(output[1])
        # Enable gradient tracking for attention matrix
        self.attention_matrix[-1].requires_grad_(True)
        # self.attention_matrix[-1].retain_grad()
        return output

    def backward_hook(self, grad):
        self.attention_grad.append(grad)


# Generate Gaussian noise to add to embeddings
def generate_gaussian_noise(input_embeddings, mean=0.0, std_dev=0.1):
    noise = torch.normal(mean=mean, std=std_dev, size=input_embeddings.size())
    return noise


def GAD(model, tokenizer, template, prompt, theta, candidates):

    input_embeddings, before_embeddings, after_embeddings, _ = get_embeddings(model, tokenizer, template, prompt)

    start = 0.0
    stop = 0.5
    step = (stop - start) / (candidates - 1)

    batch = []
    for i in range(candidates):
        std = start + i * step
        noise = generate_gaussian_noise(input_embeddings, mean=0.0, std_dev=std)[0]
        noise = torch.abs(noise).to(model.device)

        noise_embeddings = input_embeddings + noise
        # Adjust embedding data type
        noise_embeddings = noise_embeddings.to(dtype=model.get_input_embeddings().weight.dtype)

        input_embeds = torch.cat([
            before_embeddings,
            noise_embeddings,
            after_embeddings,
        ], dim=1)
        batch.append(input_embeds)

    batch = torch.cat(batch, dim=0)

    outputs = model.generate(
        inputs_embeds=batch,
        max_new_tokens=256,
        do_sample=False,
        return_dict_in_generate=True,
        use_cache=False
    )
    response = outputs['sequences']

    all_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in response]

    # Reference from: https://github.com/arobey1/smooth-llm/blob/main/lib/defenses.py
    # Check whether the outputs jailbreak the LLM
    are_copies_jailbroken = [is_jailbroken(s) for s in all_outputs]
    if len(are_copies_jailbroken) == 0:
        raise ValueError("LLM did not generate any outputs.")

    outputs_and_jbs = zip(all_outputs, are_copies_jailbroken)

    # Determine whether SGD was jailbroken
    jb_percentage = np.mean(are_copies_jailbroken)
    is_jb = True if jb_percentage > theta else False

    # Pick a response that is consistent with the majority vote
    majority_outputs = [
        output for (output, jb) in outputs_and_jbs
        if jb == is_jb
    ]

    if is_jb:
        response = majority_outputs[0]
    else:
        response = random.choice(majority_outputs)

    del outputs, input_embeddings, before_embeddings, after_embeddings, _, noise_embeddings, input_embeds
    gc.collect()

    return response, jb_percentage


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="baichuan2_7b", help="The name of the model to use")
    parser.add_argument("--Jailbreak", type=str, default="FewShot", help="The name of the Jailbreak to use")
    parser.add_argument("--theta", type=validate_theta, default=0.9, help="Voting thresholds")
    parser.add_argument("--candidates", type=int, default=10, help="Number of candidates")

    args = parser.parse_args()

    model_name = args.model_name
    Jailbreak = args.Jailbreak
    candidates = args.candidates
    theta = args.theta

    adversarial_prompts_path = "adversarial_prompts/{}/{}/results/{}.json".format(Jailbreak, model_name, model_name)

    adversarial_prompts = read_json_file(adversarial_prompts_path)

    model, tokenizer = load_model_and_tokenizer(model_path[model_name], "bf16", device_map="balanced_low_0", trust_remote_code=True)
    template = get_template(model_path[model_name])

    # Load or create defense results file
    defense_results_file = "defense_results/GradShield_ablation_grad_{}_{}.json".format(Jailbreak, model_name)
    defense_results = read_json_file(defense_results_file)

    if not defense_results:
        print(
            "{} file does not exist, a new defense_results.json file will be created".format(defense_results_file))
        defense_results = {}

    count = 0
    start_time = time.time()
    for key, value in tqdm(adversarial_prompts.items()):
        if key in defense_results:
            continue
        for item in value:
            if item.get("label") == 1:
                count += 1
                prompt = item.get("test_case")

                response, jb_percentage = GAD(model, tokenizer, template, prompt, theta, candidates)

                # Store response in defense results
                defense_results[key] = {
                    "prompt": prompt,
                    "response": response,
                    "refuse_attention": [],
                    "jb_percentage": jb_percentage,
                    "label": None
                }

                # Write defense results to JSON file
                with open(defense_results_file, "w") as file:
                    json.dump(defense_results, file, indent=4)

                model.zero_grad()
                torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run: {elapsed_time}?")
