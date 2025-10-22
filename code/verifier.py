import argparse
import json
import re
import os
import gc
from typing import Dict
from vllm import LLM, SamplingParams


# -------------------- Utility Functions --------------------

def load_json(file_path: str) -> list:
    """Load JSON data from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: list, file_path: str) -> None:
    """Save JSON data to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_json(text: str) -> Dict:
    """Attempt to extract a JSON object from a string with several fallback methods."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback 1: extract first JSON-like block
    matches = re.findall(r'{(.*?)}', text, re.DOTALL)
    for match in matches:
        candidate = '{' + match + '}'
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Fallback 2: normalize quotes and keys
    text = text.replace("'", '"')
    text = re.sub(r'(\w+):', r'"\1":', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# -------------------- Prompt Construction --------------------

def build_verification_prompt(question: str, options: list, answer: str, cot_content: str) -> str:
    """Construct a verification prompt for Chain-of-Thought (CoT) validation."""
    options_str = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    return f"""You are a medical reasoning verification expert. Determine whether the following Chain-of-Thought (CoT) reasoning correctly supports the provided answer.

[Question]
{question}

[Options]
{options_str}

[Correct Answer]
{answer}

[CoT Analysis]
{cot_content}

Evaluate the reasoning according to these criteria:
1. Does it correctly identify key clinical features?
2. Are all options properly evaluated?
3. Does the logic lead to the correct answer?
4. Are there any factual medical errors?

Output a JSON object with the following structure:
{{
  "verdict": "Correct" | "Error",
  "reason": "Brief explanation (1-2 sentences)."
}}
"""


# -------------------- Main Processing Functions --------------------

def batch_inference(args, llm) -> None:
    """Perform batch verification for all CoT fields and store intermediate results."""
    data = load_json(args.input_json)
    tasks = []
    cot_pattern = re.compile(r'^model\d+_COT\d+$')

    for index, item in enumerate(data):
        base_info = {k: v for k, v in item.items() if not cot_pattern.match(k)}
        for key, value in item.items():
            if cot_pattern.match(key):
                prompt = build_verification_prompt(
                    question=item.get("question", ""),
                    options=item.get("options", []),
                    answer=item.get("answer", ""),
                    cot_content=value
                )
                tasks.append({
                    "item_index": index,
                    "cot_field": key,
                    "cot_content": value,
                    "base_info": base_info,
                    "raw_prompt": prompt
                })

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</json>"]
    )

    prompts = [t["raw_prompt"] for t in tasks]
    outputs = llm.generate(prompts, sampling_params)
    for task, output in zip(tasks, outputs):
        task["llm_response"] = output.outputs[0].text.strip()

    save_json(tasks, args.intermediate_file)


def iterative_verification(args, llm) -> None:
    """Iteratively process intermediate verification results, retrying invalid ones."""
    tasks = load_json(args.intermediate_file)
    if not tasks:
        print("No intermediate data found.")
        return

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</json>"]
    )

    # Parse existing valid results
    for task in tasks:
        json_resp = extract_json(task.get("llm_response", ""))
        if json_resp and "verdict" in json_resp and "reason" in json_resp:
            task["verdict"] = "Correct" if json_resp["verdict"].strip().lower() == "correct" else "Error"
            task["verification_reason"] = json_resp["reason"]

    # Retry invalid tasks
    pending = [t for t in tasks if "verdict" not in t]
    iteration = 0
    while pending and iteration < args.max_iterations:
        print(f"Iteration {iteration + 1}: Retrying {len(pending)} tasks...")
        outputs = llm.generate([t["raw_prompt"] for t in pending], sampling_params)
        for task, out in zip(pending, outputs):
            resp = out.outputs[0].text.strip()
            task["llm_response"] = resp
            json_resp = extract_json(resp)
            if json_resp and "verdict" in json_resp and "reason" in json_resp:
                task["verdict"] = "Correct" if json_resp["verdict"].strip().lower() == "correct" else "Error"
                task["verification_reason"] = json_resp["reason"]
        pending = [t for t in tasks if "verdict" not in t]
        iteration += 1

    # Mark any remaining invalid tasks
    for t in pending:
        t["verdict"] = "Error"
        t["verification_reason"] = "Exceeded maximum retry attempts."

    save_json([], args.intermediate_file)

    # Split into correct/error results
    correct_individual, error_individual = [], []
    correct_aggregated, error_aggregated = {}, {}

    for task in tasks:
        idx = task["item_index"]
        if task["verdict"].lower() == "correct":
            correct_individual.append(task)
            correct_aggregated.setdefault(idx, task["base_info"].copy())
            correct_aggregated[idx][task["cot_field"]] = task["cot_content"]
        else:
            error_individual.append(task)
            error_aggregated.setdefault(idx, task["base_info"].copy())
            error_aggregated[idx][task["cot_field"]] = task["cot_content"]

    save_json(correct_individual, args.correct_output)
    save_json(error_individual, args.error_output)
    save_json(list(correct_aggregated.values()), args.correct_split_output)
    save_json(list(error_aggregated.values()), args.error_split_output)


# -------------------- Main Entry --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative CoT Verification and Categorization Pipeline")
    parser.add_argument("--input_json", required=True, help="Input JSON containing CoT fields.")
    parser.add_argument("--model_path", required=True, help="Path to the model used for verification.")
    parser.add_argument("--intermediate_file", default="intermediate_results.json", help="Intermediate output file.")
    parser.add_argument("--correct_output", default="Other/correct_0cot_reason.json", help="Individual correct CoT results.")
    parser.add_argument("--error_output", default="Other/error_0cot_reason.json", help="Individual error CoT results.")
    parser.add_argument("--correct_split_output", default="correct_0cots.json", help="Aggregated correct CoTs.")
    parser.add_argument("--error_split_output", default="error_0cots.json", help="Aggregated error CoTs.")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Tensor parallel size for inference.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--max_tokens", type=int, default=10000, help="Maximum token generation length.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Max retry iterations for invalid JSON outputs.")

    args = parser.parse_args()

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)

    if not os.path.exists(args.intermediate_file) or os.path.getsize(args.intermediate_file) == 0:
        batch_inference(args, llm)

    iterative_verification(args, llm)

    del llm
    gc.collect()
