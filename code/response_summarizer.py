import argparse
import json
import os
import gc
import time
from typing import List, Dict, Any
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm
import concurrent.futures  # 用于多线程并发处理

def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def make_prompt(cot: str) -> str:
    return f"""Summarize the following chain-of-thought reasoning:
{cot}
"""

# 处理请求函数，添加重试机制
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def process_request_with_retry(client, model, prompt, max_tokens):
    try:
        messages = [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except (RetryError, Exception) as e:
        print(f"Warning: error occurred during API request: {e}")
        if '429' in str(e):  # If it's a 429 error, wait before retrying
            print("Rate limit exceeded, retrying after a delay...")
            time.sleep(60)  # Wait 60 seconds before retrying
            return process_request_with_retry(client, model, prompt, max_tokens)  # Recursive retry
        return None

def transform(args):
    data = load_json(args.input_json)
    prompts = [make_prompt(item.get("output", "").strip()) for item in data]

    # Initialize OpenAI API client
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.api_key,
        api_version="2024-10-21"
    )

    results = []

    # 使用 tqdm 显示进度条
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        # 提交所有请求任务
        futures = {
            executor.submit(process_request_with_retry, client, args.model, prompt, args.max_tokens): idx
            for idx, prompt in enumerate(prompts)
        }

        # 处理每个请求的结果
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing", unit="item"):
            idx = futures[future]
            summary = future.result()
            if summary is not None:
                results.append({
                    "instruction": data[idx].get("instruction", ""),
                    "input": data[idx].get("input", ""),
                    "COT": data[idx].get("output", ""),
                    "Response": summary
                })
            else:
                print(f"Failed to process item {idx}, skipping.")

    save_json(results, args.results_file)
    print(f"Saved {len(results)} summaries to {args.results_file}")

    del client
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize CoT outputs into one-sentence responses")
    parser.add_argument("--input_json", required=True, help="Input JSON with instruction/input/output fields")
    parser.add_argument("--results_file", default="results.json", help="Output JSON file with summaries")
    parser.add_argument("--model", required=True, help="OpenAI model name.")
    parser.add_argument("--azure_endpoint", required=True, help="Azure OpenAI API endpoint.")
    parser.add_argument("--api_key", required=True, help="Azure OpenAI API key.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum tokens to generate per summary")
    args = parser.parse_args()

    transform(args)