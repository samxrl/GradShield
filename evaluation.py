import torch
from uitils import get_adversarial_prompt_model_name, read_json_file, model_path
from model_utils import get_template, load_model_and_tokenizer
from GradShield import GradShield
import argparse
import json
from tqdm import tqdm


def parse_jailbreak_list(values):
    raw_value = " ".join(values)
    normalized_value = raw_value.replace("[", " ").replace("]", " ").replace(",", " ")
    return [item.strip("'\"") for item in normalized_value.split()]


def run_evaluation_for_jailbreak(model, tokenizer, template, model_name, jailbreak):
    adversarial_prompt_model_name = get_adversarial_prompt_model_name(model_name)
    if adversarial_prompt_model_name != model_name:
        print("Using adversarial prompts from {} for {}".format(adversarial_prompt_model_name, model_name))

    adversarial_prompts_path = "adversarial_prompts/{}/{}/results/{}.json".format(
        jailbreak,
        adversarial_prompt_model_name,
        adversarial_prompt_model_name,
    )

    adversarial_prompts = read_json_file(adversarial_prompts_path)
    if not adversarial_prompts:
        print("{} file does not exist or is empty, skipping {}".format(adversarial_prompts_path, jailbreak))
        return

    # Load or create defense results file
    defense_results_file = "defense_results/defense_results_{}_{}.json".format(jailbreak, model_name)
    defense_results = read_json_file(defense_results_file)

    if not defense_results:
        print(
            "{} file does not exist, a new defense_results.json file will be created".format(defense_results_file))
        defense_results = {}

    print("Running {} on {}".format(jailbreak, model_name))
    for key, value in tqdm(adversarial_prompts.items()):
        if key in defense_results:
            continue
        for item in value:
            # Check if the item is a successful jailbreak attack
            if item.get("label") == 1:
                prompt = item.get("test_case")

                response, token_importance = GradShield(model, tokenizer, template, prompt, copies=10 , std=(0.05, 0.5), top_k=4)

                # Store response in defense results
                defense_results[key] = {
                    "prompt": prompt,
                    "response": response,
                    "token_importance": token_importance.tolist(),
                    "label": None
                }

                # Write defense results to JSON file
                with open(defense_results_file, "w") as file:
                    json.dump(defense_results, file, indent=4)

                model.zero_grad()
                torch.cuda.empty_cache()


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-8B", help="The name of the LLM")
    parser.add_argument(
        "--Jailbreak",
        nargs="+",
        default=["GCG","PAIR","TAP","AutoPrompt","FewShot"],
        help="The names of Jailbreak methods, for example: --Jailbreak GCG PAIR or --Jailbreak \"[GCG,PAIR]\"",
    )

    args = parser.parse_args()

    model_name = args.model_name
    jailbreaks = parse_jailbreak_list(args.Jailbreak)
    if not jailbreaks:
        parser.error("--Jailbreak must contain at least one jailbreak method")

    model, tokenizer = load_model_and_tokenizer(model_path[model_name], "bf16", device_map="balanced_low_0", trust_remote_code=True)
    template = get_template(model_path[model_name])

    for jailbreak in jailbreaks:
        run_evaluation_for_jailbreak(model, tokenizer, template, model_name, jailbreak)
