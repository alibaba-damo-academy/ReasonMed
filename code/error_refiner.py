import argparse
import json
import os
import re
import gc

from vllm import LLM, SamplingParams


def load_json(file_path):
    """Safely load a JSON file, returning an empty list if the file doesn't exist or isn't valid JSON."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return []


def save_json(data, file_path):
    """Write data to a JSON file."""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_cot(text: str) -> str:
    """Remove redundant markers and extra whitespace."""
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



def main(args):
    data = load_json(args.input_json)
    if not data:
        print(f"Failed to read valid data from {args.input_json}.")
        return

    # Prepare LLM and sampling params
    sampling = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=30000)
    llm = LLM(model=args.model_path, tensor_parallel_size=8)

    # Aggregate prompts for all items
    tasks = []
    prompts = []
    for item in data:
        question = item.get("question", "")
        options = item.get("options", [])
        options_str = "\n- " + "\n- ".join(options)
        answer = item.get("answer", "")
        reasons_map = item.get("verification_reasons", {})
        # Iterate over all COT fields
        for cot_field, original_cot in item.items():
            if re.match(r'model\d+_COT\d+', cot_field):
                # Collect error reasons from other COTs
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

    # Generate a list of sampling params corresponding to each prompt, keeping the same length
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

    # Clean up and save
    del llm
    gc.collect()

    save_json(final_list, args.output_json)
    print(f"Optimized COTs have been saved to {args.output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch optimize COTs in one go')
    parser.add_argument('--input_json', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the vLLM model')
    parser.add_argument('--output_json', type=str, required=True, help='Path to the output JSON file')
    args = parser.parse_args()
    main(args)
