import argparse
import json
import os
import re

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm

# Constants for prompt cleaning
CLEAN_PATTERN_THINKING = r"##\s*Thinking[\s\S]*?\n"
CLEAN_PATTERN_TAGS = r"</?think>"
CLEAN_PATTERN_ALRIGHT = r"^\s*Alright\s*$"

# Template for optimization prompt
template = '''
You are an expert clinician-educator AI tutor. Your mission is to generate an exceptionally comprehensive, in-depth chain-of-thought explanation that rigorously justifies the correct answer for the given clinical MCQ, while specifically addressing and integrating provided error feedback to eliminate previous reasoning flaws. Adhere closely to these instructions to maximize completeness:

1. **Deep Comprehension**  
   - Thoroughly parse the question stem: patient demographics, clinical history, symptoms, signs, labs, imaging, and all answer choices.  
   - Highlight any subtle findings or “red flag” details.

2. **Error-Driven Refinement**  
   - Review the provided **Error Reasons from Other Attempts**.  
   - Identify logical gaps, factual mistakes, omissions, or misleading inferences in the original chain‐of‐thought.  
   - Explicitly incorporate corrections and clarifications derived from these error reasons.

3. **Structured, Layered Reasoning**  
Organize your explanation into clear sections:
   a) Restate the question in your own words.  
   b) Highlight the key clinical details and relevant background information (e.g., pathophysiology, anatomy, typical presentations, diagnostic tests).
   c) Evaluate each candidate answer, discussing supporting evidence and potential pitfalls.
   d) Systematically rule out options that do not align with the clinical context.
   e) Compare any remaining choices based on their merits.
   f) Conclude with your final answer accompanied by a clear and concise summary of your reasoning.

4. **Depth & Detail**  
   - Articulate each point in complete sentences, weaving clinical context and rationale rather than presenting terse statements.  
   - Embed quantitative metrics—prevalence, sensitivity, specificity, likelihood ratios, Number Needed to Treat (NNT), and relative risks—within pre- and post-test probability frameworks to clarify diagnostic and therapeutic impact.  
   - Cite landmark guidelines or pivotal studies and integrate epidemiologic modifiers (e.g., age, sex, comorbidities) to ground statements in evidence and tailor patient-specific nuances.

5. **Polish & Coherence**  
   - After drafting, refine the language for clarity, flow, and professional tone.  
   - Use transition phrases to guide the reader through your logic.

**Inputs**  
- **Question:**  `{question}`  
- **Options:**  `{options}`  
- **Correct Answer:**  `{answer}`  
- **Original Chain-of-Thought:**  `{original_cot}`  
- **Error Reasons from Other Attempts:**  `{error_reasons}`  

**Output:**  
Please output *only* the final, optimized chain-of-thought explanation as plain text, with no additional headings or formatting. Ensure that you explicitly address and rectify each error reason provided.
'''

def load_json(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return []


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_cot(text: str) -> str:
    text = re.sub(CLEAN_PATTERN_THINKING, "", text)
    text = re.sub(CLEAN_PATTERN_TAGS, "", text)
    text = re.sub(CLEAN_PATTERN_ALRIGHT, "", text, flags=re.MULTILINE)
    return text.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_openai(client, model, messages, **params):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **params
    )


def main():
    parser = argparse.ArgumentParser(description='Batch optimize COTs via Azure OpenAI API')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save optimized COT JSON')
    parser.add_argument('--usage_json', type=str, required=True, help='Path to save prompt/completion usage JSON')
    parser.add_argument('--model', type=str, default='o1-mini-0912', help='Model name (e.g., o1-mini-0912')
    parser.add_argument('--azure_endpoint', type=str, default='', help='Azure endpoint URL')
    parser.add_argument('--api_key', type=str, required=True, help='Azure OpenAI API key')
    parser.add_argument('--api_version', type=str, default='2024-10-21', help='Azure OpenAI API version')
    args = parser.parse_args()

    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.api_key,
        api_version=args.api_version
    )

    data = load_json(args.input_json)
    if not data:
        print(f"No valid records loaded from {args.input_json}.")
        return

    optimized_records = []
    usage_records = []

    # Wrap the iteration with tqdm for progress display
    for item in tqdm(data, desc="Optimizing COTs", unit="item"):
        question = item.get("question", "")
        options = item.get("options", [])
        options_str = "\n- " + "\n- ".join(options)
        answer = item.get("answer", "")
        reasons_map = item.get("verification_reasons", {})

        # Locate original COT
        original_cot = None
        cot_field = None
        correct_map = item.get("correct", {})
        if isinstance(correct_map, dict):
            for k, v in correct_map.items():
                if re.match(r'model\d+_COT\d+', k) and isinstance(v, str):
                    original_cot = v
                    cot_field = k
                    break
        if original_cot is None:
            for k, v in item.items():
                if re.match(r'model\d+_COT\d+', k) and isinstance(v, str):
                    original_cot = v
                    cot_field = k
                    break
        if original_cot is None:
            tqdm.write(f"Warning: no COT field found for question: {question}")
            continue

        error_reasons = [f"- {v}" for mk, v in reasons_map.items() if mk != cot_field and v]
        prompt_text = template.format(
            question=question,
            options=options_str,
            answer=answer,
            original_cot=original_cot,
            error_reasons="\n".join(error_reasons)
        )

        messages = [
            {"role": "user", "content": prompt_text}
        ]

        try:
            resp = call_openai(
                client,
                args.model,
                messages,
                max_completion_tokens=20000,
                stop=None
            )
        except RetryError as retry_err:
            tqdm.write(f"Warning: skipping item due to API error: {retry_err.last_attempt.exception()}")
            continue
        except Exception as e:
            tqdm.write(f"Warning: skipping item due to unexpected error: {e}")
            continue

        optimized = clean_cot(resp.choices[0].message.content)
        optimized_records.append({
            "question": question,
            "options": options,
            "answer": answer,
            "model_choice": cot_field,
            "COT": optimized,
            "difficulty": item.get("difficulty", "")
        })

        usage_records.append({
            "question": question,
            "model_choice": cot_field,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens
        })

    save_json(optimized_records, args.output_json)
    save_json(usage_records, args.usage_json)

    print(f"Optimized COT saved to {args.output_json}")
    print(f"Token usage saved to {args.usage_json}")

if __name__ == '__main__':
    main()
