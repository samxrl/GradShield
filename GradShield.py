import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from uitils import is_refused
from torch import Tensor
import numpy as np
import torch.nn.functional as F
import gc
import random


DEFAULT_INSTRUCTION_PLACEHOLDER = "{instruction}"


def tokenize_without_special_tokens(tokenizer: AutoTokenizer, text: str, device):
    if text == "":
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def embed_token_ids(model: AutoModelForCausalLM, token_ids: Tensor):
    embedding_layer = model.get_input_embeddings()
    if token_ids.shape[1] == 0:
        return embedding_layer.weight.new_empty((1, 0, embedding_layer.embedding_dim))
    with torch.no_grad():
        return embedding_layer(token_ids)


def normalize_segment(segment):
    if isinstance(segment, dict):
        return segment["placeholder"], segment["text"]
    if isinstance(segment, (tuple, list)) and len(segment) == 2:
        return segment[0], segment[1]
    raise ValueError("User segment must be a dict or a (placeholder, text) pair.")


def get_user_segments(template, test_case_input: str):
    segments = [
        normalize_segment(segment)
        for segment in template.get("user_segments", [])
    ]
    current_placeholder = template.get("current_user_placeholder", DEFAULT_INSTRUCTION_PLACEHOLDER)
    segments.append((current_placeholder, test_case_input))
    return segments


def split_template_prompt(prompt_template: str, segments):
    parts = []
    offset = 0
    for placeholder, _ in segments:
        placeholder_index = prompt_template.find(placeholder, offset)
        if placeholder_index == -1:
            raise ValueError("Placeholder '{}' was not found in the prompt template.".format(placeholder))
        parts.append(prompt_template[offset:placeholder_index])
        offset = placeholder_index + len(placeholder)
    parts.append(prompt_template[offset:])
    return parts


def get_prompt_embeddings(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, template, test_case_input: str):
    segments = get_user_segments(template, test_case_input)
    prompt_parts = split_template_prompt(template["prompt"], segments)

    embedding_chunks = []
    perturb_mask = []
    user_tokens = []

    for index, (_, segment_text) in enumerate(segments):
        part_ids = tokenize_without_special_tokens(tokenizer, prompt_parts[index], model.device)
        part_embeddings = embed_token_ids(model, part_ids)
        embedding_chunks.append(part_embeddings)
        perturb_mask.extend([0.0] * part_ids.shape[1])

        segment_ids = tokenize_without_special_tokens(tokenizer, segment_text, model.device)
        segment_embeddings = embed_token_ids(model, segment_ids)
        embedding_chunks.append(segment_embeddings)
        perturb_mask.extend([1.0] * segment_ids.shape[1])
        user_tokens.extend(tokenizer.convert_ids_to_tokens(segment_ids[0].tolist()))

    final_part_ids = tokenize_without_special_tokens(tokenizer, prompt_parts[-1], model.device)
    final_part_embeddings = embed_token_ids(model, final_part_ids)
    embedding_chunks.append(final_part_embeddings)
    perturb_mask.extend([0.0] * final_part_ids.shape[1])

    input_embeddings = torch.cat(embedding_chunks, dim=1)
    perturb_mask = np.array(perturb_mask, dtype=np.float32)
    if not np.any(perturb_mask):
        raise ValueError("No user tokens were found for GradShield perturbation.")

    return input_embeddings, perturb_mask, user_tokens


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


def get_perplexity(model: AutoModelForCausalLM, input_embeds: Tensor, target_ids: Tensor):
    outputs = model(inputs_embeds=input_embeds, output_attentions=True)
    logits = outputs.logits

    # Select logits corresponding to target tokens
    logits = logits[:, -target_ids.size(-1) - 1:-1, :]

    # Calculate cross-entropy loss
    loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

    # Compute perplexity from loss
    perplexity = torch.exp(loss_ce)

    return -perplexity


# Hook to capture attention gradients
class AttentionGradientHook:
    def __init__(self):
        self.attention_matrix = []
        self.attention_grad = []

    def forward_hook(self, module, input, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
            return output
        # Capture attention matrix from forward pass
        self.attention_matrix.append(output[1])
        # Enable gradient tracking for attention matrix
        self.attention_matrix[-1].requires_grad_(True)
        return output

    def backward_hook(self, grad):
        self.attention_grad.append(grad)


def normalize_token_importance(weighted_attentions, perturb_mask):
    token_importance = np.zeros_like(weighted_attentions, dtype=np.float32)
    user_positions = perturb_mask > 0
    user_attentions = np.maximum(weighted_attentions[user_positions], 0)

    min_val = np.min(user_attentions)
    max_val = np.max(user_attentions)
    if max_val != min_val:
        token_importance[user_positions] = (user_attentions - min_val) / (max_val - min_val)

    return token_importance


def gradient_weighted_attention(model: AutoModelForCausalLM, tokenizer, template, test_case_input: str, top_k):
    input_embeds, perturb_mask, _ = get_prompt_embeddings(model, tokenizer, template, test_case_input)
    prompt_token_count = input_embeds.shape[1]

    outputs = model.generate(
        inputs_embeds=input_embeds,
        max_new_tokens=top_k,
        do_sample=False,
        return_dict_in_generate=True,
    )

    prefix = outputs['sequences'][0].unsqueeze(0)
    prefix_embeds = model.get_input_embeddings()(prefix)

    # Initialize attention hook
    hook = AttentionGradientHook()

    # Register forward hooks on each model layer
    forward_handles = []
    try:
        for layer in model.model.layers:
            forward_handle = layer.register_forward_hook(hook.forward_hook)
            forward_handles.append(forward_handle)

        input_embeds = torch.cat([
            input_embeds,
            prefix_embeds
        ], dim=1)

        model.zero_grad()
        perplexity = get_perplexity(model, input_embeds, prefix)

        if not hook.attention_matrix:
            raise RuntimeError("No attention matrices were captured for gradient weighting.")

        # Retrieve captured attention matrix
        attention_matrix = hook.attention_matrix

        # Compute gradients of attention matrix
        attention_grad = torch.autograd.grad(perplexity, attention_matrix, retain_graph=True)

        weighted_attentions = torch.zeros_like(attention_matrix[0][0])
        for attention, grad in zip(attention_matrix, attention_grad):
            weighted_attention = grad[0] * attention[0]
            weighted_attentions += weighted_attention.to(weighted_attentions.device)

        weighted_attentions = torch.sum(weighted_attentions, dim=(0, 1))

        weighted_attentions = weighted_attentions.detach().to(torch.float).cpu().numpy()
        weighted_attentions = weighted_attentions[:prompt_token_count]
        token_importance = normalize_token_importance(weighted_attentions, perturb_mask)

        return token_importance, perturb_mask
    finally:
        for forward_handle in forward_handles:
            forward_handle.remove()

        model.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()


# Generate Gaussian noise to add to embeddings
def generate_gaussian_noise(input_embeddings, mean=0.0, std_dev=0.1):
    noise = torch.normal(mean=mean, std=std_dev, size=input_embeddings.size())
    return noise


def GradShield(model, tokenizer, template, prompt, copies=10, std=(0.05,0.5) ,top_k=4):

    token_importance, perturb_mask = gradient_weighted_attention(model, tokenizer, template, prompt, top_k)
    input_embeddings, _, _ = get_prompt_embeddings(model, tokenizer, template, prompt)

    min_std = std[0]
    max_std = std[1]
    step = (max_std - min_std) / (copies - 1) if copies > 1 else 0

    batch = []
    for i in range(copies):
        current_std = min_std + i * step
        noise = generate_gaussian_noise(input_embeddings, mean=0.0, std_dev=current_std)[0]
        noise = torch.abs(noise).to(model.device)
        mask = torch.tensor(token_importance).view(noise.shape[0], 1).to(model.device)

        noise = noise * mask
        noise_embeddings = input_embeddings + noise
        # Adjust embedding data type
        noise_embeddings = noise_embeddings.to(dtype=model.get_input_embeddings().weight.dtype)

        batch.append(noise_embeddings)

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

    # Check whether the outputs jailbreak the LLM
    are_copies_are_refused = [is_refused(s) for s in all_outputs]
    if len(are_copies_are_refused) == 0:
        raise ValueError("LLM did not generate any outputs.")

    outputs_and_refuse = zip(all_outputs, are_copies_are_refused)


    is_ref = True if True in are_copies_are_refused else False

    # Pick a response that is consistent with the majority vote
    majority_outputs = [
        output for (output, refused) in outputs_and_refuse
        if refused == is_ref
    ]

    if not is_ref:
        response = majority_outputs[0]
    else:
        response = random.choice(majority_outputs)

    returned_token_importance = token_importance[perturb_mask > 0]

    del outputs, input_embeddings, noise_embeddings, batch
    gc.collect()

    return response, returned_token_importance
