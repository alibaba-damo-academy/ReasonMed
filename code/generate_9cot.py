import argparse
import json
import os
import gc

from vllm import LLM, SamplingParams

def load_json(file_path):
    """安全加载 JSON 文件，若文件不存在或非 JSON 格式则返回空列表。"""
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
    """将 data 写入 JSON 文件。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_text(output_obj):
    """安全提取 vLLM 输出对象中的生成文本。"""
    if hasattr(output_obj, "outputs") and output_obj.outputs:
        return output_obj.outputs[0].text.strip()
    elif hasattr(output_obj, "text"):
        return output_obj.text.strip()
    else:
        return ""

def main(args):
    # ================= 0. 加载数据 =================
    # 如果中间文件存在，则从中间文件加载数据，续跑后续模型，否则从原始数据加载
    if args.intermediate_path and os.path.exists(args.intermediate_path):
        data = load_json(args.intermediate_path)
        print(f"已加载中间文件 {args.intermediate_path}，将继续后续推理。")
    else:
        data = load_json(args.data_path)
        if not data:
            print(f"无法从 {args.data_path} 读取有效数据，请检查文件")
            return

    # 定义 3 组采样参数（各生成 3 个 CoT）
    sampling_params_list_model = [
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

    # 优化后的单条 CoT 生成提示模板
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

Please provide your detailed chain-of-thought reasoning followed by your final answer.
"""

    # ================= 1. 使用 Model1 生成 3 个 CoT =================
    if not data[0].get("model1_COT1"):
        print(">>> [1/3] Loading Model1 and generating 3 CoTs for each sample ...")
        llm_model1 = LLM(
            model=args.model_path1,
            tensor_parallel_size=8
        )
        all_prompts_model1 = []
        question_indices_model1 = []
        for idx, item in enumerate(data):
            question = item.get("question", "")
            options_list = item.get("options", [])
            options_str = "\n- " + "\n- ".join(options_list)
            for _ in sampling_params_list_model:
                prompt = single_cot_prompt_template.format(question=question, options=options_str)
                all_prompts_model1.append(prompt)
                question_indices_model1.append(idx)
        all_sampling_params_model1 = sampling_params_list_model * len(data)
        outputs_model1 = llm_model1.generate(all_prompts_model1, all_sampling_params_model1)
        idx_map = {}
        for i, output_obj in enumerate(outputs_model1):
            q_idx = question_indices_model1[i]
            idx_map[q_idx] = idx_map.get(q_idx, 0)
            text = extract_text(output_obj)
            data[q_idx][f"model1_COT{idx_map[q_idx] + 1}"] = text
            idx_map[q_idx] += 1
        del llm_model1
        gc.collect()
        print(">>> Model1 inference finished, 3 CoTs per sample have been generated.\n")
        # 保存第一阶段中间结果
        if args.intermediate_path:
            save_json(data, args.intermediate_path)
            print(f"已写入第一阶段的中间结果 -> {args.intermediate_path}")
    else:
        print("Model1 CoTs 已存在，跳过第一阶段生成。")

    # ================= 2. 使用 Model2 生成 3 个 CoT =================
    if not data[0].get("model2_COT1"):
        print(">>> [2/3] Loading Model2 and generating 3 CoTs for each sample ...")
        llm_model2 = LLM(
            model=args.model_path2,
            tensor_parallel_size=8
        )
        all_prompts_model2 = []
        question_indices_model2 = []
        for idx, item in enumerate(data):
            question = item.get("question", "")
            options_list = item.get("options", [])
            options_str = "\n- " + "\n- ".join(options_list)
            for _ in sampling_params_list_model:
                prompt = single_cot_prompt_template.format(question=question, options=options_str)
                all_prompts_model2.append(prompt)
                question_indices_model2.append(idx)
        all_sampling_params_model2 = sampling_params_list_model * len(data)
        outputs_model2 = llm_model2.generate(all_prompts_model2, all_sampling_params_model2)
        idx_map2 = {}
        for i, output_obj in enumerate(outputs_model2):
            q_idx = question_indices_model2[i]
            idx_map2[q_idx] = idx_map2.get(q_idx, 0)
            text = extract_text(output_obj)
            data[q_idx][f"model2_COT{idx_map2[q_idx] + 1}"] = text
            idx_map2[q_idx] += 1
        del llm_model2
        gc.collect()
        print(">>> Model2 inference finished, 3 CoTs per sample have been generated.\n")
        if args.intermediate_path:
            save_json(data, args.intermediate_path)
            print(f"已写入第二阶段的中间结果 -> {args.intermediate_path}")
    else:
        print("Model2 CoTs 已存在，跳过第二阶段生成。")

    # ================= 3. 使用 Model3 生成 3 个 CoT =================
    if not data[0].get("model3_COT1"):
        print(">>> [3/3] Loading Model3 and generating 3 CoTs for each sample ...")
        llm_model3 = LLM(
            model=args.model_path3,
            tensor_parallel_size=8
        )
        all_prompts_model3 = []
        question_indices_model3 = []
        for idx, item in enumerate(data):
            question = item.get("question", "")
            options_list = item.get("options", [])
            options_str = "\n- " + "\n- ".join(options_list)
            for _ in sampling_params_list_model:
                prompt = single_cot_prompt_template.format(question=question, options=options_str)
                all_prompts_model3.append(prompt)
                question_indices_model3.append(idx)
        all_sampling_params_model3 = sampling_params_list_model * len(data)
        outputs_model3 = llm_model3.generate(all_prompts_model3, all_sampling_params_model3)
        idx_map3 = {}
        for i, output_obj in enumerate(outputs_model3):
            q_idx = question_indices_model3[i]
            idx_map3[q_idx] = idx_map3.get(q_idx, 0)
            text = extract_text(output_obj)
            data[q_idx][f"model3_COT{idx_map3[q_idx] + 1}"] = text
            idx_map3[q_idx] += 1
        del llm_model3
        gc.collect()
        print(">>> Model3 inference finished, 3 CoTs per sample have been generated.\n")
        # —— 将模型3的中间结果追加写入，而不是覆盖之前内容
        if args.intermediate_path:
            save_json(data, args.intermediate_path)
            print(f"已写入第三阶段的中间结果 -> {args.intermediate_path}")
    else:
        print("Model3 CoTs 已存在，跳过第三阶段生成。")

    # ================= 最终结果保存 =================
    save_json(data, args.json_path)
    print(f">>> Done! The final results with all CoTs have been saved to {args.json_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="MedLLM Inference Pipeline (Single Command) with 3 Models")
    parser.add_argument("--data_path", type=str, default="test.json",
                        help="原始输入数据（包含问题+选项）的 JSON 文件。必须是列表格式。")
    parser.add_argument("--model_path1", type=str, default="/path/to/model1",
                        help="第一个模型的路径，用于生成前 3 个 CoT。")
    parser.add_argument("--model_path2", type=str, default="/path/to/model2",
                        help="第二个模型的路径，用于生成中间 3 个 CoT。")
    parser.add_argument("--model_path3", type=str, default="/path/to/model3",
                        help="第三个模型的路径，用于生成最后 3 个 CoT。")
    parser.add_argument("--json_path", type=str, default="final_result.json",
                        help="最终输出文件，用于保存包含原始数据和9条 CoT 的推理结果。")
    parser.add_argument("--intermediate_path", type=str, default="",
                        help="可选：若指定，会在各阶段结果生成后将中间数据写入此 JSON 文件，用于排查或调试。")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)