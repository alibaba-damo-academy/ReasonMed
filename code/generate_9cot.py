import argparse
import json
import os
import re
import gc

from vllm import LLM, SamplingParams

# Define functions to load and save JSON data

def load_json(file_path):
    """Safely load a JSON file, returning an empty list if the file doesn't exist or isn't valid JSON."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except json.JSONDecodeError:
        return []


def save_json(data, file_path):
    """Write data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_text(output_obj):
    """Safely extract generated text from a vLLM output object."""
    if hasattr(output_obj, "outputs") and output_obj.outputs:
        return output_obj.outputs[0].text.strip()
    elif hasattr(output_obj, "text"):
        return output_obj.text.strip()
    else:
        return ""


def main(args):
    # ================= 0. Load data =================
    # If an intermediate file exists, load data from it to resume inference; otherwise, load from the original data file
    if args.intermediate_path and os.path.exists(args.intermediate_path):
        data = load_json(args.intermediate_path)
        print(f"Loaded intermediate file {args.intermediate_path}, resuming downstream inference.")
    else:
        data = load_json(args.data_path)
        if not data:
            print(f"Unable to read valid data from {args.data_path}. Please check the file.")
            return

    # Define three sampling parameter sets (to generate three CoTs each)
    sampling_params_list = [
        SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=8192,
            stop=[]
        ),
        SamplingParams(
            temperature=0.9,
            top_p=0.9,
            max_tokens=8192,
            stop=[]
        ),
        SamplingParams(
            temperature=1.0,
            top_p=0.9,
            max_tokens=8192,
            stop=[]
        )
    ]

    # Prompt template for generating a single chain of thought
    single_cot_prompt_template = """You are a highly knowledgeable medical expert. You are provided with a clinical multiple-choice question along with several candidate answers.
Your task is to carefully analyze the clinical scenario and each option by following these steps:
1. Restate the question in your own words.
2. Highlight the key clinical details and relevant background information (e.g., pathophysiology, anatomy, typical presentations, diagnostic tests).
3. Evaluate each candidate answer, discussing supporting evidence and potential pitfalls.
4. Systematically rule out options that do not align with the clinical context.
5. Compare any remaining choices based on their merits.
6. Conclude with your final answer accompanied by a clear and concise summary of your reasoning.

Please note: Your response should be based solely on the current question and candidate answers. Do not consider any previous context or prior interactions.

Question:
{question}

Candidate Answers:
{options}

Please provide your detailed chain-of-thought reasoning followed by your final answer."""

    # ================= 1. Generate 3 CoTs using Model1 =================
    if not data[0].get("model1_COT1"):
        print(">>> [1/3] Loading Model1 and generating 3 CoTs for each sample ...")
        llm1 = LLM(
            model=args.model_path1,
            tensor_parallel_size=8
        )
        prompts1 = []
        idx_map1 = []
        for idx, item in enumerate(data):
            question = item.get("question", "")
            options_list = item.get("options", [])
            options_str = "\n- " + "\n- ".join(options_list)
            for _ in sampling_params_list:
                prompts1.append(single_cot_prompt_template.format(question=question, options=options_str))
                idx_map1.append(idx)
        params1 = sampling_params_list * len(data)
        outputs1 = llm1.generate(prompts1, params1)
        count1 = {}
        for i, out in enumerate(outputs1):
            qidx = idx_map1[i]
            count1[qidx] = count1.get(qidx, 0) + 1
            text = extract_text(out)
            data[qidx][f"model1_COT{count1[qidx]}"] = text
        del llm1
        gc.collect()
        print(">>> Model1 inference finished. Generated 3 CoTs per sample.\n")
        if args.intermediate_path:
            save_json(data, args.intermediate_path)
            print(f"Wrote phase-1 intermediate results -> {args.intermediate_path}")
    else:
        print("Model1 CoTs already exist, skipping phase 1.")

    # ================= 2. Generate 3 CoTs using Model2 =================
    if not data[0].get("model2_COT1"):
        print(">>> [2/3] Loading Model2 and generating 3 CoTs for each sample ...")
        llm2 = LLM(
            model=args.model_path2,
            tensor_parallel_size=8
        )
        prompts2 = []
        idx_map2 = []
        for idx, item in enumerate(data):
            question = item.get("question", "")
            options_list = item.get("options", [])
            options_str = "\n- " + "\n- ".join(options_list)
            for _ in sampling_params_list:
                prompts2.append(single_cot_prompt_template.format(question=question, options=options_str))
                idx_map2.append(idx)
        params2 = sampling_params_list * len(data)
        outputs2 = llm2.generate(prompts2, params2)
        count2 = {}
        for i, out in enumerate(outputs2):
            qidx = idx_map2[i]
            count2[qidx] = count2.get(qidx, 0) + 1
            text = extract_text(out)
            data[qidx][f"model2_COT{count2[qidx]}"] = text
        del llm2
        gc.collect()
        print(">>> Model2 inference finished. Generated 3 CoTs per sample.\n")
        if args.intermediate_path:
            save_json(data, args.intermediate_path)
            print(f"Wrote phase-2 intermediate results -> {args.intermediate_path}")
    else:
        print("Model2 CoTs already exist, skipping phase 2.")

    # ================= 3. Generate 3 CoTs using Model3 =================
    if not data[0].get("model3_COT1"):
        print(">>> [3/3] Loading Model3 and generating 3 CoTs for each sample ...")
        llm3 = LLM(
            model=args.model_path3,
            tensor_parallel_size=8
        )
        prompts3 = []
        idx_map3 = []
        for idx, item in enumerate(data):
            question = item.get("question", "")
            options_list = item.get("options", [])
            options_str = "\n- " + "\n- ".join(options_list)
            for _ in sampling_params_list:
                prompts3.append(single_cot_prompt_template.format(question=question, options=options_str))
                idx_map3.append(idx)
        params3 = sampling_params_list * len(data)
        outputs3 = llm3.generate(prompts3, params3)
        count3 = {}
        for i, out in enumerate(outputs3):
            qidx = idx_map3[i]
            count3[qidx] = count3.get(qidx, 0) + 1
            text = extract_text(out)
            data[qidx][f"model3_COT{count3[qidx]}"] = text
        del llm3
        gc.collect()
        print(">>> Model3 inference finished. Generated 3 CoTs per sample.\n")
        if args.intermediate_path:
            save_json(data, args.intermediate_path)
            print(f"Wrote phase-3 intermediate results -> {args.intermediate_path}")
    else:
        print("Model3 CoTs already exist, skipping phase 3.")

    # ================= Final results saving =================
    save_json(data, args.json_path)
    print(f">>> Done! The final results with all CoTs have been saved to {args.json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MedLLM Inference Pipeline (Single Command) with 3 Models")
    parser.add_argument("--data_path", type=str, default="test.json",
                        help="Original input JSON file containing questions and options. Must be a list.")
    parser.add_argument("--model_path1", type=str, default="/path/to/model1",
                        help="Path to the first model for generating the first 3 CoTs.")
    parser.add_argument("--model_path2", type=str, default="/path/to/model2",
                        help="Path to the second model for generating the next 3 CoTs.")
    parser.add_argument("--model_path3", type=str, default="/path/to/model3",
                        help="Path to the third model for generating the final 3 CoTs.")
    parser.add_argument("--json_path", type=str, default="final_result.json",
                        help="Output JSON file to save the original data with 9 CoT results.")
    parser.add_argument("--intermediate_path", type=str, default="",
                        help="Optional: if provided, intermediate data will be written to this JSON file after each phase for debugging.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
