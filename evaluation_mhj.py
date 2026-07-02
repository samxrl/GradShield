import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm

from GradShield import GradShield
from mhj_utils import load_mhj_records, summarize_mhj_records
from model_utils import get_template, load_model_and_tokenizer
from uitils import model_path, read_json_file


MAX_EVALUATED_USER_TURNS = 15


def remove_bos_token(tokenizer, prompt):
    if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
        return prompt.replace(tokenizer.bos_token, "", 1)
    return prompt


def qwen3_chat_template_kwargs(model_name_or_path):
    normalized_model_path = (model_name_or_path or "").replace("\\", "/").lower()
    if "qwen3" in normalized_model_path:
        return {"enable_thinking": False}
    return {}


def apply_chat_template(tokenizer, model_name_or_path, messages):
    kwargs = qwen3_chat_template_kwargs(model_name_or_path)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def render_plain_history(history):
    lines = []
    role_labels = {
        "system": "System",
        "user": "USER",
        "assistant": "ASSISTANT",
    }
    for message in history:
        role = role_labels.get(message["role"], message["role"])
        lines.append("{}: {}".format(role, message["content"]))

    if not lines:
        return ""

    return "\n".join(lines) + "\nUSER: "


def fallback_turn_template(base_template, history):
    before_str, after_str = base_template["prompt"].split("{instruction}", 1)
    return {
        "description": base_template.get("description", "") + " with MHJ history",
        "prompt": before_str + render_plain_history(history) + "{instruction}" + after_str,
    }


def build_turn_template(tokenizer, base_template, model_name_or_path, history):
    messages = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in history
    ]
    messages.append({"role": "user", "content": "{instruction}"})

    try:
        prompt = apply_chat_template(tokenizer, model_name_or_path, messages)
        prompt = remove_bos_token(tokenizer, prompt)
        if "{instruction}" not in prompt:
            raise ValueError("chat template dropped the instruction placeholder")

        return {
            "description": "MHJ multi-turn template from tokenizer.apply_chat_template",
            "prompt": prompt,
        }
    except Exception:
        return fallback_turn_template(base_template, history)


def defense_results_path(model_name):
    return Path("defense_results") / "defense_results_MHJ_{}.json".format(model_name)


def no_defense_results_path(model_name):
    return Path("defense_results") / "defense_results_MHJ_no_defense_{}.json".format(model_name)


def get_results_path(model_name, defense):
    if defense == "none":
        return no_defense_results_path(model_name)
    return defense_results_path(model_name)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)


def generate_without_defense(model, tokenizer, template, prompt, max_new_tokens=256):
    full_prompt = template["prompt"].replace("{instruction}", prompt)
    encoded = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    output_ids = outputs[:, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_record(
        model,
        tokenizer,
        base_template,
        model_name_or_path,
        record,
        copies,
        std,
        top_k,
        save_turn_importance,
        defense,
):
    history = []
    turns = []
    final_response = ""
    final_token_importance = None
    user_turn_index = 0

    for message in record["messages"]:
        if message["role"] == "system":
            history.append({"role": "system", "content": message["body"]})
            continue

        if message["role"] != "user":
            history.append({"role": message["role"], "content": message["body"]})
            continue

        user_turn_index += 1
        turn_template = build_turn_template(
            tokenizer,
            base_template,
            model_name_or_path,
            history,
        )

        if defense == "none":
            response = generate_without_defense(model, tokenizer, turn_template, message["body"])
            token_importance = None
        else:
            response, token_importance = GradShield(
                model,
                tokenizer,
                turn_template,
                message["body"],
                copies=copies,
                std=std,
                top_k=top_k,
            )

        turn = {
            "turn_index": user_turn_index,
            "user_message": message["body"],
            "response": response,
            "label": None,
        }
        final_response = response
        final_token_importance = token_importance.tolist() if token_importance is not None else None

        if final_token_importance is not None and (save_turn_importance or user_turn_index == record["user_turn_count"]):
            turn["token_importance"] = final_token_importance

        turns.append(turn)
        history.append({"role": "user", "content": message["body"]})
        history.append({"role": "assistant", "content": response})

        model.zero_grad()
        torch.cuda.empty_cache()

    return {
        "behavior_id": record["behavior_id"],
        "question_id": record["question_id"],
        "source": record["source"],
        "tactic": record["tactic"],
        "temperature": record["temperature"],
        "user_turn_count": record["user_turn_count"],
        "message_count": record["message_count"],
        "defense": defense,
        "prompt": turns[-1]["user_message"] if turns else "",
        "response": final_response,
        "token_importance": final_token_importance,
        "turns": turns,
        "label": None,
    }


def run_mhj_evaluation(args):
    records = load_mhj_records(
        args.mhj_csv,
        args.harmbench_csv,
        scope=args.scope,
    )
    skipped_by_turn_count = [
        record for record in records
        if record["user_turn_count"] > MAX_EVALUATED_USER_TURNS
    ]
    records = [
        record for record in records
        if record["user_turn_count"] <= MAX_EVALUATED_USER_TURNS
    ]
    if skipped_by_turn_count:
        skipped_ids = ", ".join(record["mhj_id"] for record in skipped_by_turn_count)
        print(
            "Warning: skipping {} MHJ records with more than {} user turns: {}".format(
                len(skipped_by_turn_count),
                MAX_EVALUATED_USER_TURNS,
                skipped_ids,
            )
        )

    if args.max_samples is not None:
        records = records[:args.max_samples]

    summary = summarize_mhj_records(records)
    print("Loaded MHJ records: {}".format(json.dumps(summary, ensure_ascii=False)))

    model_name_or_path = model_path[args.model_name]
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path,
        "bf16",
        device_map="balanced_low_0",
        trust_remote_code=True,
    )
    base_template = get_template(model_name_or_path)

    results_file = get_results_path(args.model_name, args.defense)
    defense_results = read_json_file(str(results_file))
    if not defense_results:
        print("{} file does not exist, a new file will be created".format(results_file))
        defense_results = {}

    print("Running MHJ ({}) on {}".format(args.defense, args.model_name))
    for record in tqdm(records):
        mhj_id = record["mhj_id"]
        if mhj_id in defense_results:
            continue

        try:
            defense_results[mhj_id] = run_record(
                model,
                tokenizer,
                base_template,
                model_name_or_path,
                record,
                copies=args.copies,
                std=(args.std_min, args.std_max),
                top_k=args.top_k,
                save_turn_importance=args.save_turn_importance,
                defense=args.defense,
            )
            write_json(results_file, defense_results)
        except Exception as exc:
            warning = (
                "Warning: skipping {} (behavior_id={}, user_turn_count={}) "
                "because {}: {}"
            ).format(
                mhj_id,
                record["behavior_id"],
                record["user_turn_count"],
                exc.__class__.__name__,
                exc,
            )
            tqdm.write(warning)
            model.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-8B", help="The name of the LLM")
    parser.add_argument(
        "--mhj_csv",
        type=str,
        default="adversarial_prompts/MHJ_harmbench_behaviors.csv",
        help="Path to MHJ_harmbench_behaviors.csv",
    )
    parser.add_argument(
        "--harmbench_csv",
        type=str,
        default="adversarial_prompts/harmbench_behaviors_text_all.csv",
        help="Path to HarmBench behavior CSV",
    )
    parser.add_argument(
        "--scope",
        choices=["multi_turn", "all"],
        default="multi_turn",
        help="Use only multi-turn conversations by default, or all MHJ conversations",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--copies", type=int, default=10, help="Number of perturbed copies")
    parser.add_argument("--std_min", type=float, default=0.05, help="Minimum Gaussian noise std")
    parser.add_argument("--std_max", type=float, default=0.5, help="Maximum Gaussian noise std")
    parser.add_argument("--top_k", type=int, default=4, help="Top-k generated tokens for importance")
    parser.add_argument(
        "--defense",
        choices=["gradshield", "none"],
        default="gradshield",
        help="Use GradShield or run a no-defense baseline",
    )
    parser.add_argument(
        "--save_turn_importance",
        action="store_true",
        help="Save token importance for every user turn instead of only the final turn",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_mhj_evaluation(parse_args())
