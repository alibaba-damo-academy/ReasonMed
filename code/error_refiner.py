import argparse
import json
import os
import re
import gc

from vllm import LLM, SamplingParams


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


def clean_cot(text: str) -> str:
    """移除冗余标记和多余空白。"""
    text = re.sub(r"##\s*Thinking[\s\S]*?\n", "", text)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"^\s*Alright\s*$", "", text, flags=re.MULTILINE)
    return text.strip()

OPTIMIZE_PROMPT = '''
You are an expert clinician-educator AI tutor. Your mission is to generate an exceptionally comprehensive, in-depth chain-of-thought explanation that rigorously justifies the correct answer for the given clinical MCQ, while specifically addressing and integrating provided error feedback to eliminate previous reasoning flaws. Adhere closely to these instructions to maximize completeness:

1. **Error-Driven Refinement**  
   - Review the provided **Error Reasons from Other Attempts**.  
   - Identify logical gaps, factual mistakes, omissions, or misleading inferences in the original chain‐of‐thought.  
   - Explicitly incorporate corrections and clarifications derived from these error reasons.

2. **Structured, Layered Reasoning**  
Organize your explanation into clear sections:
   a) **Restated Question** – Summarize the clinical scenario in your own words.  
   b) **Key Details** – Enumerate and elaborate on all critical findings.  
   c) **Pathophysiologic Basis** – Explain underlying mechanisms linking findings to disease processes.  
   d) **Option-by-Option Analysis** – For each choice:  
      - Describe why it could be considered.  
      - Cite clinical guidelines, studies, or pathophysiology.  
      - Explain why it is ultimately correct or incorrect, referencing corrections from error feedback when relevant.  
   e) **Differential Comparison** – Contrast the two most plausible options, noting subtle distinctions and how error insights resolve ambiguity.  
   f) **Final Justification** – State the correct answer and succinctly summarize why it outperforms all alternatives.  
   g) **Clinical Pearls** – Offer relevant mnemonics, guideline references, or high-yield take-home points.


**Inputs**   
- **Question:**  '{question}'  
- **Options:**  '{options}'  
- **Correct Answer:**  '{answer}'  
- **Original Chain-of-Thought:**  '{original_cot}'  
- **Error Reasons from Other Attempts:**  '{error_reasons}'  

**Output:**  
Please optimized Original Chain-of-Thought. Ensure that you explicitly address and rectify each error reason provided.
'''


# 提示模板
# OPTIMIZE_PROMPT = '''
# You are an expert AI tutor. You are given a clinical multiple-choice question, one original chain-of-thought (COT) explanation, and a list of error reasons from other COT attempts. Your task is to produce one refined and error-free chain-of-thought explanation that correctly justifies the answer. Please:
# 1. Review the question and answer choices thoroughly.
# 2. Use the provided error reasons to correct any logical gaps, factual inaccuracies, or reasoning oversights in the original COT.
# 3. Remove any redundant tokens or tags such as "## Thinking", "Alright", "<think>", or "</think>".
# 4. Structure your reasoning in a detailed, step-by-step manner:
#    a) Restate the clinical question in your own words.
#    b) Identify and elaborate on key clinical details (e.g., patient demographics, symptoms, lab results).
#    c) Explain the underlying pathophysiology and how it applies to this scenario.
#    d) Evaluate each option individually, noting why it may or may not be appropriate.
#    e) Eliminate incorrect options with clear, evidence-based justification.
#    f) Compare the remaining options, highlighting subtle differences.
#    g) Conclude with the final answer, summarizing why it is the best choice.
# 5. After outlining the reasoning, refine the explanation for clarity, coherence, ensuring smooth language flow.

# Question:
# {question}

# Options:
# {options}

# Answer:
# {answer}

# Original COT:
# {original_cot}

# Error Reasons (from other COTs):
# {error_reasons}

# Please output only the optimized COT text without additional formatting.
# '''  


def main(args):
    data = load_json(args.input_json)
    if not data:
        print(f"无法从 {args.input_json} 读取有效数据。")
        return

    # Prepare LLM and sampling params
    sampling = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=30000)
    llm = LLM(model=args.model_path, tensor_parallel_size=8)

    # 聚合所有 item 的 prompts
    tasks = []
    prompts = []
    for item in data:
        question = item.get("question", "")
        options = item.get("options", [])
        options_str = "\n- " + "\n- ".join(options)
        answer = item.get("answer", "")
        reasons_map = item.get("verification_reasons", {})
        # 遍历所有 COT 字段
        for cot_field, original_cot in item.items():
            if re.match(r'model\d+_COT\d+', cot_field):
                # 收集其他 COT 的错误原因
                error_reasons = [f"- {v}" for k, v in reasons_map.items() if k != cot_field and v]
                prompt = OPTIMIZE_PROMPT.format(
                    question=question,
                    options=options_str,
                    answer=answer,
                    original_cot=original_cot,
                    error_reasons="\n".join(error_reasons)
                )
                tasks.append({
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "difficulty": item.get("difficulty", ""),
                    "cot_field": cot_field
                })
                prompts.append(prompt)

    # 为每个 prompt 生成对应的 sampling 参数列表，长度保持一致
    params_list = [sampling] * len(prompts)
    outputs = llm.generate(prompts, params_list)

    final_list = []
    for task, out in zip(tasks, outputs):
        text = out.outputs[0].text.strip()
        optimized = clean_cot(text)
        final_list.append({
            "question": task["question"],
            "options": task["options"],
            "answer": task["answer"],
            "model_choice": task["cot_field"],
            "COT": optimized,
            "difficulty": task.get("difficulty", "")
        })

    # 清理并保存
    del llm
    gc.collect()

    save_json(final_list, args.output_json)
    print(f"优化后的 COT 已保存至 {args.output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch optimize COTs in one go')
    parser.add_argument('--input_json', type=str, required=True, help='输入 JSON 文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='vLLM 模型路径')
    parser.add_argument('--output_json', type=str, required=True, help='输出 JSON 文件路径')
    args = parser.parse_args()
    main(args)