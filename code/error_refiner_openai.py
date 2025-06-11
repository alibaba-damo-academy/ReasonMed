import argparse
import json
import os
import re
import gc
import time
from tqdm import tqdm
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import concurrent.futures
from openai import AzureOpenAI  # 导入 AzureOpenAI 库

# 定义函数加载和保存 JSON 数据
def load_json(file_path):
    """安全加载 JSON 文件，若文件不存在或非 JSON 格式则返回空列表。"""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return []

def save_json(data, file_path):
    """将 data 写入 JSON 文件。"""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# 清理 COT 内容，移除冗余的标记和空白
def clean_cot(text: str) -> str:
    """移除冗余标记和多余空白。"""
    text = re.sub(r"##\s*Thinking[\s\S]*?\n", "", text)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"^\s*Alright\s*$", "", text, flags=re.MULTILINE)
    return text.strip()

# 提示模板
OPTIMIZE_PROMPT = '''
You are an expert clinician-educator AI tutor. Your mission is to generate an exceptionally comprehensive, in-depth chain-of-thought explanation that rigorously justifies the correct answer for the given clinical MCQ, while specifically addressing and integrating provided error feedback to eliminate previous reasoning flaws. Adhere closely to these instructions to maximize completeness:

1. **Error-Driven Refinement**  
   - Review the provided **Error Reasons from Other Attempts**.  
   - Identify logical gaps, factual mistakes, omissions, or misleading inferences in the original chain‐of‐thought.  
   - Explicitly incorporate corrections and clarifications derived from these error reasons.

2. **Structured, Layered Reasoning**  
Organize your explanation into clear sections:
a. Restate the question in your own words.
b. Highlight the key clinical details and relevant background information (e.g., pathophysiology, anatomy, typical presentations, diagnostic tests).
c. Evaluate each candidate answer, discussing supporting evidence and potential pitfalls.
d. Systematically rule out options that do not align with the clinical context.
e. Compare any remaining choices based on their merits.
f. Conclude with your final answer accompanied by a clear and concise summary of your reasoning.



**Inputs**   
- **Question:**  '{question}'  
- **Options:**  '{options}'  
- **Correct Answer:**  '{answer}'  
- **Original Chain-of-Thought:**  '{original_cot}'  
- **Error Reasons from Other Attempts:**  '{error_reasons}'  

**Output:**  
Please optimized Original Chain-of-Thought. Ensure that you explicitly address and rectify each error reason provided.
'''

# 使用 Azure OpenAI API 进行推理，并添加重试机制
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def azure_api_request(client, prompt: str, model: str, max_tokens: int) -> str:
    """通过 Azure OpenAI API 发送请求并返回响应结果"""
    try:
        messages = [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8,
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except (RetryError, Exception) as e:
        print(f"API 请求错误: {e}")
        if '429' in str(e):  # 如果是 429 错误，等待重试
            print("Rate limit exceeded, retrying after a delay...")
            time.sleep(60)  # 等待 60 秒后重试
            return azure_api_request(client, prompt, model, max_tokens)  # 递归重试
        return ""

# 使用多进程进行并发处理
def main(args):
    data = load_json(args.input_json)
    if not data:
        print(f"无法从 {args.input_json} 读取有效数据。")
        return

    # 设置 Azure OpenAI API 客户端
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.api_key,
        api_version="2024-10-21"
    )

    final_list = []

    # 使用 concurrent.futures 进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        futures = {}

        for idx, item in enumerate(data):
            question = item.get("question", "")
            options = item.get("options", [])
            answer = item.get("answer", "")
            reasons_map = item.get("verification_reasons", {})

            # 遍历所有 COT 字段
            for cot_field, original_cot in item.items():
                if re.match(r'model\d+_COT\d+', cot_field):
                    # 收集其他 COT 的错误原因
                    error_reasons = [f"- {v}" for k, v in reasons_map.items() if k != cot_field and v]
                    prompt = OPTIMIZE_PROMPT.format(
                        question=question,
                        options="\n- " + "\n- ".join(options),
                        answer=answer,
                        original_cot=original_cot,
                        error_reasons="\n".join(error_reasons)
                    )

                    # 提交任务
                    futures[executor.submit(azure_api_request, client, prompt, args.model, args.max_tokens)] = idx

        # 处理结果
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing", unit="item"):
            idx = futures[future]
            optimized_cot = future.result()
            if optimized_cot:
                optimized = clean_cot(optimized_cot)
                final_list.append({
                    "question": data[idx].get("question", ""),
                    "options": data[idx].get("options", []),
                    "answer": data[idx].get("answer", ""),
                    "model_choice": list(data[idx].keys())[0],  # 假设每个条目只有一个 COT 字段
                    "COT": optimized,
                    "difficulty": data[idx].get("difficulty", "")
                })

    # 清理并保存
    save_json(final_list, args.output_json)
    print(f"优化后的 COT 已保存至 {args.output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch optimize COTs in one go using Azure OpenAI API')
    parser.add_argument('--input_json', type=str, required=True, help='输入 JSON 文件路径')
    parser.add_argument('--api_key', type=str, required=True, help='Azure OpenAI API 密钥')
    parser.add_argument('--azure_endpoint', type=str, required=True, help='Azure OpenAI API endpoint')
    parser.add_argument('--model', type=str, required=True, help='Azure OpenAI 模型名称')
    parser.add_argument('--output_json', type=str, required=True, help='输出 JSON 文件路径')
    parser.add_argument('--max_tokens', type=int, default=15000, help='最大生成 tokens 数量')
    args = parser.parse_args()
    main(args)