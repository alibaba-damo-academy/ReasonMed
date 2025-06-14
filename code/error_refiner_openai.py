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
from openai import AzureOpenAI  # Import AzureOpenAI library

# Define functions to load and save JSON data
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

# Clean up chain-of-thought content by removing redundant markers and whitespace
def clean_cot(text: str) -> str:
    """Remove redundant markers and extra whitespace."""
    text = re.sub(r"##\s*Thinking[\s\S]*?\n", "", text)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"^\s*Alright\s*$", "", text, flags=re.MULTILINE)
    return text.strip()

# Prompt template for optimization tasks
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
Please optimize the provided chain-of-thought, explicitly addressing and rectifying each error reason."'''

# Use Azure OpenAI API for inference with retry mechanism
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def azure_api_request(client, prompt: str, model: str, max_tokens: int) -> str:
    """Send a request through the Azure OpenAI API and return the response content."""
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
        print(f"API request error: {e}")
        if '429' in str(e):  # Rate limit error handling
            print("Rate limit exceeded, retrying after a delay...")
            time.sleep(60)  # Wait 60 seconds before retry
            return azure_api_request(client, prompt, model, max_tokens)
        return ""

# Main function: batch processing with concurrency
def main(args):
    data = load_json(args.input_json)
    if not data:
        print(f"Unable to read valid data from {args.input_json}.")
        return

    # Initialize Azure OpenAI API client
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.api_key,
        api_version="2024-10-21"
    )

    final_list = []

    # Process items concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        futures = {}

        for idx, item in enumerate(data):
            question = item.get("question", "")
            options = item.get("options", [])
            answer = item.get("answer", "")
            reasons_map = item.get("verification_reasons", {})

            # Iterate over all COT fields
            for cot_field, original_cot in item.items():
                if re.match(r'model\d+_COT\d+', cot_field):
                    # Collect error reasons from other attempts
                    error_reasons = [f"- {v}" for k, v in reasons_map.items() if k != cot_field and v]
                    prompt = OPTIMIZE_PROMPT.format(
                        question=question,
                        options="\n- " + "\n- ".join(options),
                        answer=answer,
                        original_cot=original_cot,
                        error_reasons="\n".join(error_reasons)
                    )

                    futures[executor.submit(azure_api_request, client, prompt, args.model, args.max_tokens)] = (idx, cot_field)

        # Process completed futures
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing", unit="item"):
            idx, cot_field = futures[future]
            optimized_cot = future.result()
            if optimized_cot:
                optimized = clean_cot(optimized_cot)
                final_list.append({
                    "question": data[idx].get("question", ""),
                    "options": data[idx].get("options", []),
                    "answer": data[idx].get("answer", ""),
                    "model_choice": cot_field,
                    "COT": optimized,
                    "difficulty": data[idx].get("difficulty", "")
                })

    # Save the optimized results
    save_json(final_list, args.output_json)
    print(f"Optimized COTs have been saved to {args.output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch optimize COTs in one go using Azure OpenAI API')
    parser.add_argument('--input_json', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--api_key', type=str, required=True, help='Azure OpenAI API key')
    parser.add_argument('--azure_endpoint', type=str, required=True, help='Azure OpenAI API endpoint')
    parser.add_argument('--model', type=str, required=True, help='Azure OpenAI model name')
    parser.add_argument('--output_json', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--max_tokens', type=int, default=15000, help='Maximum number of tokens to generate')
    args = parser.parse_args()
    main(args)
