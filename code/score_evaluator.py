import argparse
import json
import os
import random
import re
import gc
import concurrent.futures  # 并发请求处理
import time  # 用于处理重试时的延时
from typing import List, Dict, Any
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm  # 导入tqdm

# 加载JSON文件的函数
def load_json(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 保存JSON文件的函数
def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# 提取JSON的函数
def extract_json(text: str) -> Dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for match in re.findall(r"{.*?}", text, flags=re.S):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    fixed = text.replace("'", '"')
    fixed = re.sub(r"(\w+)\s*:", r'"\1":', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None

# 评分Prompt
def scoring_prompt(question: str, answer: str, response: str) -> str:
    return f"""You are a medical reasoning evaluator. Assess the following response based on the following criteria:

1. **Clinical accuracy**: Does the response correctly incorporate medical facts, clinical guidelines, and evidence-based practices? Are the clinical details provided accurate, relevant, and appropriate for the given situation?
2. **Logical reasoning**: Does the response logically follow the reasoning process required to arrive at the answer? Is the reasoning chain coherent and well-supported by evidence or clinical knowledge?
3. **Factual correctness**: Are there any factual errors in the response? Are all statements factually correct and consistent with established medical knowledge?
4. **Completeness**: Does the response cover all necessary aspects of the question? Is it thorough and detailed, addressing the key points without missing critical information?

[Question]
{question}

[Response]
{response}

Please evaluate the response on the above criteria and provide a JSON object with two keys:
  "score": integer between 1 and 10 (1 = poor reasoning, significant factual or clinical errors, incomplete or illogical response; 10 = flawless reasoning, no factual errors, highly accurate and complete response).
  "justification": A concise explanation of your score, addressing the clinical accuracy, logical reasoning, factual correctness, and completeness.
"""

# 请求处理函数，添加了重试机制
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def process_request_with_retry(client, model, t, max_tokens):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{t['prompt']}"}
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["</json>"]
        )
        raw_response = resp.choices[0].message.content.strip()
        js = extract_json(raw_response)
        score = 0
        if isinstance(js, dict) and "score" in js and isinstance(js["score"], int):
            score = max(1, min(10, js["score"]))
        return score
    except (RetryError, Exception) as e:
        print(f"Warning: error occurred for item {t['item_index']}: {e}")
        if '429' in str(e):  # 如果是429错误，等待后重试
            print("Rate limit exceeded, retrying after a delay...")
            time.sleep(60)  # 等待60秒
            return process_request_with_retry(client, model, t, max_tokens)  # 递归调用进行重试
        elif 'NoneType' in str(e):  # 防止 NoneType 错误
            print(f"Error with item {t['item_index']}: NoneType response")
        return None

# 批量评分处理
def batch_score(args: argparse.Namespace, client: AzureOpenAI, input_json: str, intermediate_file: str) -> None:
    data = load_json(input_json)

    # 随机抽取3000个样本进行推理
    sampled_data = random.sample(data, 3000)  # 抽取3000个样本
    tasks: list[dict] = []

    for idx, item in enumerate(sampled_data):
        question = item.get("instruction", "")
        response = item.get("output", "")
        answer = item.get("input", "")

        prompt = scoring_prompt(question, answer, response)
        tasks.append({
            "item_index": idx,
            "question": question,
            "answer": answer,
            "response": response,
            "prompt": prompt
        })

    # 使用并发处理推理请求
    scores = []
    successful_samples = 0
    failed_samples = 0  # Track failed samples
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # 调整并发数
        futures = {executor.submit(process_request_with_retry, client, args.model, t, args.max_tokens): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tasks", unit="task"):
            score = future.result()
            if score is not None:
                scores.append(score)
                successful_samples += 1
            else:
                failed_samples += 1  # Increment failed samples

            # 如果已成功处理3000个样本，停止执行
            if successful_samples >= 3000:
                print(f"Successfully processed {successful_samples} samples. Stopping further processing.")
                break

    # Log the failed samples count
    print(f"Total failed samples: {failed_samples}")

    # 在成功处理后保存评分结果
    if successful_samples >= 3000:
        print("Saving the scores to file...")
        save_json(scores, intermediate_file)

        # 计算并输出3000个样本的平均分
        avg_score = sum(scores) / len(scores)
        print(f"Average score for 3000 samples: {avg_score:.2f}")
    else:
        print("Insufficient valid samples processed, not saving the result.")

# 计算平均分数
def calculate_average_score(final_output: str) -> float:
    if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
        data = load_json(final_output)
        if len(data) == 0:
            return 0
        return sum(data) / len(data)
    else:
        print(f"Final output file {final_output} is either missing or empty. Skipping average score calculation.")
        return 0

# 主程序入口
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Assess reasoning quality and score responses.")
    p.add_argument("--input_jsons", required=True, nargs='+', help="Input files with full reasoning data.")
    p.add_argument("--model", required=True, help="OpenAI model name.")
    p.add_argument("--azure_endpoint", required=True, help="Azure OpenAI API endpoint.")
    p.add_argument("--api_key", required=True, help="Azure OpenAI API key.")
    p.add_argument("--intermediate_file", default="intermediate_scores.json")
    p.add_argument("--final_output", default="final_scores.json", help="Final cleaned result file.")
    p.add_argument("--max_tokens", type=int, default=4000)

    args = p.parse_args()

    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.api_key,
        api_version="2024-10-21"
    )

    for input_json in args.input_jsons:
        intermediate_file = f"intermediate_{os.path.basename(input_json)}.json"
        final_output = f"final_{os.path.basename(input_json)}.json"

        # Phase 1 – run everything once if no intermediate present or empty.
        if not os.path.exists(intermediate_file) or os.path.getsize(intermediate_file) == 0:
            batch_score(args, client, input_json, intermediate_file)

        # 计算平均分
        avg_score = calculate_average_score(intermediate_file)
        print(f"Average score for {input_json}: {avg_score:.2f}")

    gc.collect()

# python compare_openai.py
# Average score for 5000 samples: 8.17
# Average score for medical_o1_sft_formatted.json: 8.17

# Average score for 3000 samples: 8.08
# Average score for medical_o1_sft_formatted.json: 8.08

# Average score for 3000 samples: 8.61
# Average score for sft_reasoning_think.json: 8.61

# Average score for 3000 samples: 7.80
# Average score for cots89_sft_data.json: 7.80