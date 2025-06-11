import argparse
import json
import re
from typing import Dict
import os
import gc

from vllm import LLM, SamplingParams

def load_json(file_path: str) -> list:
    """Load JSON data from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: list, file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_json(text: str) -> Dict:
    """Attempt to extract JSON from text with multiple fallback methods."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    matches = re.findall(r'{(.*?)}', text, re.DOTALL)
    for match in matches:
        candidate = '{' + match + '}'
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    text = text.replace("'", '"')
    text = re.sub(r'(\w+):', r'"\1":', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def validate_cot_prompt_template(question: str, options: list, answer: str, cot_content: str) -> str:
    """Create verification prompt for CoT validation."""
    options_str = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    return f"""You are a medical evaluation expert. Analyze if the Chain-of-Thought (CoT) analysis correctly leads to the answer.

[Question]
{question}

[Options]
{options_str}

[Correct Answer]
{answer}

[CoT Analysis]
{cot_content}

Evaluate the CoT analysis following these criteria:
1. Does the analysis correctly identify key clinical factors?
2. Are all options appropriately considered and evaluated?
3. Does the reasoning logically lead to the correct answer?
4. Are there any factual errors in medical knowledge?

Output a JSON object with:
- "verdict": "Correct" if the CoT analysis is valid and reaches the correct answer, otherwise "Error"
- "reason": Brief explanation of your evaluation (1-2 sentences)

Example Correct Response:
{{
  "verdict": "Correct",
  "reason": "The analysis correctly identifies key pathophysiology and systematically eliminates incorrect options."
}}

Example Error Response:
{{
  "verdict": "Error",
  "reason": "The CoT misinterprets the mechanism of urethral obstruction and its renal effects."
}}

Your evaluation:
"""

def batch_inference(args, llm) -> None:
    """
    批量生成所有 CoT 字段的验证响应，并将任务保存到中间文件中。
    每个任务包括原始基本信息、CoT 字段、构造的 prompt 以及 LLM 输出的响应。
    """
    data = load_json(args.input_json)
    tasks = []
    cot_pattern = re.compile(r'^model\d+_COT\d+$')

    for index, item in enumerate(data):
        base_info = {k: v for k, v in item.items() if not cot_pattern.match(k)}
        for key, value in item.items():
            if cot_pattern.match(key):
                prompt = validate_cot_prompt_template(
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

    prompts = [task["raw_prompt"] for task in tasks]
    outputs = llm.generate(prompts, sampling_params)
    for task, output in zip(tasks, outputs):
        response_text = output.outputs[0].text.strip()
        task["llm_response"] = response_text

    save_json(tasks, args.intermediate_file)

def iterative_process_intermediate_file(args, llm) -> None:
    """
    迭代处理中间文件：
      - 加载中间文件中所有任务；
      - 对于未能生成正确 JSON 格式（缺少 "verdict" 和 "reason"）的任务进行重试，直至达到最大迭代次数；
      - 对于重试后仍无效的任务，标记为 "Error"；
      - 根据验证结果生成 individual（correct_output 与 error_output）以及聚合版本（correct_split_output 与 error_split_output）。
    """
    tasks = load_json(args.intermediate_file)
    if not tasks:
        print("中间文件为空，无需处理。")
        return

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</json>"]
    )

    # 初始化已验证任务：若 llm_response 中已能解析出正确 JSON 则更新 verdict 字段
    for task in tasks:
        json_resp = extract_json(task.get("llm_response", ""))
        if json_resp is not None and "verdict" in json_resp and "reason" in json_resp:
            task["verdict"] = "Correct" if json_resp["verdict"].strip().lower() == "correct" else "Error"
            task["verification_reason"] = json_resp["reason"]

    # 待重试任务：没有 "verdict" 字段的任务
    pending_tasks = [task for task in tasks if "verdict" not in task]
    iteration = 0
    while pending_tasks and iteration < args.max_iterations:
        print(f"迭代第 {iteration+1} 轮，重试 {len(pending_tasks)} 个任务。")
        prompts = [task["raw_prompt"] for task in pending_tasks]
        outputs = llm.generate(prompts, sampling_params)
        for task, output in zip(pending_tasks, outputs):
            new_response = output.outputs[0].text.strip()
            task["llm_response"] = new_response
            json_resp = extract_json(new_response)
            if json_resp is not None and "verdict" in json_resp and "reason" in json_resp:
                task["verdict"] = "Correct" if json_resp["verdict"].strip().lower() == "correct" else "Error"
                task["verification_reason"] = json_resp["reason"]
        pending_tasks = [task for task in tasks if "verdict" not in task]
        iteration += 1

    # 对于仍未获得有效输出的任务，直接标记为 "Error"
    for task in pending_tasks:
        task["verdict"] = "Error"
        task["verification_reason"] = "达到最大重试次数，仍然无法生成有效 JSON 输出。"

    # 清空中间文件（表示所有任务均已处理）
    save_json([], args.intermediate_file)

    # 根据 verdict 将任务划分为 individual 结果以及聚合版本
    correct_results = []
    error_results = []
    aggregated_correct = {}
    aggregated_error = {}

    for task in tasks:
        # 此时每个任务均应存在 "verdict"
        if task["verdict"].strip().lower() == "correct":
            correct_results.append(task)
            idx = task["item_index"]
            if idx not in aggregated_correct:
                aggregated_correct[idx] = task["base_info"].copy()
            aggregated_correct[idx][task["cot_field"]] = task["cot_content"]
        else:
            error_results.append(task)
            idx = task["item_index"]
            if idx not in aggregated_error:
                aggregated_error[idx] = task["base_info"].copy()
            aggregated_error[idx][task["cot_field"]] = task["cot_content"]

    correct_split_results = [aggregated_correct[idx] for idx in sorted(aggregated_correct.keys())]
    error_split_results = [aggregated_error[idx] for idx in sorted(aggregated_error.keys())]

    save_json(correct_results, args.correct_output)
    save_json(error_results, args.error_output)
    save_json(correct_split_results, args.correct_split_output)
    save_json(error_split_results, args.error_split_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="迭代式批量验证 CoT 响应并根据验证结果分离到 individual 及聚合版本输出"
    )
    parser.add_argument("--input_json", help="包含 CoT 路径的输入 JSON 文件", required=True)
    parser.add_argument("--model_path", help="验证模型的路径", required=True)
    parser.add_argument("--intermediate_file", default="intermediate_results.json",
                        help="用于存储中间推理结果的文件")
    parser.add_argument("--correct_output", default="Other/correct_0cot_reason.json",
                        help="individual 正确 CoT 输出文件")
    parser.add_argument("--error_output", default="Other/error_0cot_reason.json",
                        help="individual 错误 CoT 输出文件")
    parser.add_argument("--correct_split_output", default="correct_0cots.json",
                        help="聚合后的正确 CoT 输出文件")
    parser.add_argument("--error_split_output", default="error_0cots.json",
                        help="聚合后的错误 CoT 输出文件")
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                        help="模型加载的张量并行数")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度 (默认: 1.0)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p 采样参数 (默认: 0.95)")
    parser.add_argument("--max_tokens", type=int, default=10000,
                        help="生成的最大 tokens 数 (默认: 10000)")
    parser.add_argument("--max_iterations", type=int, default=100,
                        help="对无效输出任务的最大重试迭代次数 (默认: 100)")

    args = parser.parse_args()

    # 模型仅加载一次，并传入后续处理流程
    llm = LLM
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )

    # 若中间文件不存在或为空，则批量生成
    if not os.path.exists(args.intermediate_file) or os.path.getsize(args.intermediate_file) == 0:
        batch_inference(args, llm)
    # 迭代式处理中间文件
    iterative_process_intermediate_file(args, llm)

    del llm
    gc.collect()