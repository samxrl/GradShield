"""
Referenced from https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import login as hf_login

ALPACA_PROMPT = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
}
VICUNA_1_0_PROMPT = {
    "description": "Template used by Vicuna 1.0 and stable vicuna.",
    "prompt": "### Human: {instruction}\n### Assistant:",
}

VICUNA_PROMPT = {
    "description": "Template used by Vicuna.",
    "prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:",
}

OASST_PROMPT = {
    "description": "Template used by Open Assistant",
    "prompt": "<|prompter|>{instruction}<|endoftext|><|assistant|>"
}
OASST_PROMPT_v1_1 = {
    "description": "Template used by newer Open Assistant models",
    "prompt": "<|prompter|>{instruction}</s><|assistant|>"
}

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_CHAT_PROMPT = {
    "description": "Template used by Llama2 Chat",
    # "prompt": "[INST] {instruction} [/INST] "
    "prompt": "[INST] <<SYS>>\n" + LLAMA2_DEFAULT_SYSTEM_PROMPT + "\n<</SYS>>\n\n{instruction} [/INST] "
}

INTERNLM_PROMPT = {  # https://github.com/InternLM/InternLM/blob/main/tools/alpaca_tokenizer.py
    "description": "Template used by INTERNLM-chat",
    "prompt": "<|User|>:{instruction}<eoh><|Bot|>:"
}

KOALA_PROMPT = {  # https://github.com/young-geng/EasyLM/blob/main/docs/koala.md#koala-chatbot-prompts
    "description": "Template used by EasyLM/Koala",
    "prompt": "BEGINNING OF CONVERSATION: USER: {instruction} GPT:"
}

# Get from Rule-Following: cite
FALCON_PROMPT = {  # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

MPT_PROMPT = {  # https://huggingface.co/TheBloke/mpt-30B-chat-GGML
    "description": "Template used by MPT",
    "prompt": '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|><|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n''',
}

DOLLY_PROMPT = {
    "description": "Template used by Dolly",
    "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
}

OPENAI_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml",  # https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
'''
}

LLAMA2_70B_OASST_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml",  # https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

FALCON_INSTRUCT_PROMPT = {  # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

FALCON_CHAT_PROMPT = {  # https://huggingface.co/blog/falcon-180b#prompt-format
    "description": "Template used by Falcon Chat",
    "prompt": "User: {instruction}\nFalcon:",
}

ORCA_2_PROMPT = {
    "description": "Template used by microsoft/Orca-2-13b",
    "prompt": "<|im_start|>system\nYou are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"
}

MISTRAL_PROMPT = {
    "description": "Template used by Mistral Instruct",
    "prompt": "[INST] {instruction} [/INST]"
}

BAICHUAN_CHAT_PROMPT = {
    "description": "Template used by Baichuan2-chat",
    "prompt": "<reserved_106>{instruction}<reserved_107>"
}

QWEN_CHAT_PROMPT = {
    "description": "Template used by Qwen-chat models",
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
}

ZEPHYR_ROBUST_PROMPT = {
    "description": "",
    "prompt": "<|user|>\n{instruction}</s>\n<|assistant|>\n"
}

MIXTRAL_PROMPT = {
    "description": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "[INST] {instruction} [/INST]"
}


########## CHAT TEMPLATE ###########

def get_template(model_name_or_path=None, chat_template=None, system_message=None, **kwargs):
    # ===== Check for some older chat model templates ====
    if "wizard" in model_name_or_path.lower():
        TEMPLATE = VICUNA_PROMPT
    elif "vicuna" in model_name_or_path.lower():
        TEMPLATE = VICUNA_PROMPT
    elif "oasst" in model_name_or_path.lower():
        TEMPLATE = OASST_PROMPT
    elif "oasst_v1_1" in model_name_or_path.lower():
        TEMPLATE = OASST_PROMPT_v1_1
    elif "llama-2" in model_name_or_path.lower():
        TEMPLATE = LLAMA2_CHAT_PROMPT
    elif "falcon_instruct" in model_name_or_path.lower(): # falcon 7b / 40b instruct
        TEMPLATE = FALCON_INSTRUCT_PROMPT
    elif "falcon_chat" in model_name_or_path.lower(): # falcon 180B_chat
        TEMPLATE = FALCON_CHAT_PROMPT
    elif "mpt" in model_name_or_path.lower():
        TEMPLATE = MPT_PROMPT
    elif "koala" in model_name_or_path.lower():
        TEMPLATE = KOALA_PROMPT
    elif "dolly" in model_name_or_path.lower():
        TEMPLATE = DOLLY_PROMPT
    elif "internlm" in model_name_or_path.lower():
        TEMPLATE = INTERNLM_PROMPT
    elif "mistral" in model_name_or_path.lower() or "mixtral" in model_name_or_path.lower():
        TEMPLATE = MISTRAL_PROMPT
    elif "orca-2" in model_name_or_path.lower():
        TEMPLATE = ORCA_2_PROMPT
    elif "baichuan2" in model_name_or_path.lower():
        TEMPLATE = BAICHUAN_CHAT_PROMPT
    elif "qwen" in model_name_or_path.lower():
        TEMPLATE = QWEN_CHAT_PROMPT
    elif "zephyr_7b_robust" in model_name_or_path.lower():
        TEMPLATE = ZEPHYR_ROBUST_PROMPT
    else:
        # ======== Else default to tokenizer.apply_chat_template =======
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            template = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': '{instruction}'}] if system_message else [
                {'role': 'user', 'content': '{instruction}'}]
            prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            # Check if the prompt starts with the BOS token
            # removed <s> if it exist (LlamaTokenizer class usually have this) as our baselines will add these if needed later
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt.replace(tokenizer.bos_token, "")
            TEMPLATE = {'description': f"Template used by {model_name_or_path} (tokenizer.apply_chat_template)", 'prompt': prompt}
        except:
            assert TEMPLATE, f"Can't find instruction template for {model_name_or_path}, and apply_chat_template failed."

    print("Found Instruction template for", model_name_or_path)
    print(TEMPLATE)

    return TEMPLATE


########## MODEL ###########

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto"
}


def load_model_and_tokenizer(
        model_name_or_path,
        dtype='auto',
        device_map='auto',
        trust_remote_code=False,
        revision=None,
        token=None,
        use_fast_tokenizer=True,
        padding_side='left',
        legacy=False,
        pad_token=None,
        eos_token=None,
        load_in_8bit=False,
        **model_kwargs
):
    if token:
        hf_login(token=token)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 torch_dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype],
                                                 device_map=device_map,
                                                 trust_remote_code=trust_remote_code,
                                                 revision=revision,
                                                 load_in_8bit=load_in_8bit,
                                                 **model_kwargs).eval()

    # Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
        legacy=legacy,
        padding_side=padding_side,
    )
    if pad_token:
        tokenizer.pad_token = pad_token
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Tokenizer.pad_token is None, setting to tokenizer.unk_token")
        tokenizer.pad_token = tokenizer.unk_token
        print("tokenizer.pad_token", tokenizer.pad_token)

    return model, tokenizer
