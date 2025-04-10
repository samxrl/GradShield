import torch
from uitils import read_json_file, LLAMA2_CLS_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vicuna_7b_v1_5", help="The name of the LLM")
    parser.add_argument("--Jailbreak", type=str, default="GCG", help="The name of the Jailbreak")

    args = parser.parse_args()

    model_name = args.model_name
    Jailbreak = args.Jailbreak

    # Load defense results file
    defense_results_file = "defense_results/defense_results_{}_{}.json".format(Jailbreak, model_name)
    defense_results = read_json_file(defense_results_file)
    if not defense_results:
        print(
            "{} file does not exist, run evaluation.py to create it first".format(defense_results_file))
        exit(-1)

    # Load judgment model
    """
        model url: https://huggingface.co/cais/HarmBench-Llama-2-13b-cls
    """
    cls = AutoModelForCausalLM.from_pretrained("models/HarmBench-Llama-2-13b-cls", torch_dtype=torch.bfloat16,
                                               device_map="auto")
    cls_tokenizer = AutoTokenizer.from_pretrained("models/HarmBench-Llama-2-13b-cls", use_fast=False,
                                                  truncation_side="left",
                                                  padding_side="left")

    # load the original prompt
    file_path = 'adversarial_prompts/harmbench_behaviors_text_all.csv'
    harmbench_behaviors = pd.read_csv(file_path)

    for key, value in tqdm(defense_results.items()):
        if value['label'] != None:
            continue
        extracted_rows = harmbench_behaviors[harmbench_behaviors['BehaviorID'] == key]
        functional_category = extracted_rows['FunctionalCategory'].values[0]
        behavior = extracted_rows['Behavior'].values[0]

        if functional_category == 'contextual':
            context = extracted_rows['ContextString'].values[0]
            prompt = LLAMA2_CLS_PROMPT["prompt_contextual"].format(context=context,
                                                                   behavior=behavior,
                                                                   generation=value['response'])
        else:
            prompt = LLAMA2_CLS_PROMPT["prompt"].format(behavior=behavior,
                                                        generation=value['response'])

        encoded = cls_tokenizer(prompt, return_tensors='pt', padding='longest')
        with torch.no_grad():
            output_ids = cls.generate(
                **encoded.to(cls.device),
                do_sample=False,
                max_new_tokens=1,
            ).cpu()
            output_ids = output_ids[:, len(encoded.input_ids[0]):]

        completion = cls_tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        value['label'] = completion[0]

        # Write judgment results to JSON file
        with open(defense_results_file, "w") as file:
            json.dump(defense_results, file, indent=4)

        torch.cuda.empty_cache()

    # Calculate the DSR
    count = 0
    for key, value in tqdm(defense_results.items()):
        if value['label'] == "No":
            count += 1
    print("DSR to {} on {}: {}".format(Jailbreak, model_name, count / len(defense_results)))