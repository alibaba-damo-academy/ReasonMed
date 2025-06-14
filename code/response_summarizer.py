import argparse
import json
import os
import gc
import time
from typing import List, Dict, Any
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm
import concurrent.futures  # Used for multithreaded concurrent processing


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON data from a file, raising FileNotFoundError if the file doesn't exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: str) -> None:
    """Save data as JSON to the specified file path."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_prompt(cot: str) -> str:
    """Create a summarization prompt for a given chain-of-thought reasoning."""
    return f"""Summarize the following chain-of-thought reasoning:
{cot}
"""

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def process_request_with_retry(
    client, model: str, prompt: str, max_tokens: int
) -> Any:
    """Send a summarization request to the Azure OpenAI API with retry logic."""
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
        print(f"Warning: error during API request: {e}")
        if '429' in str(e):
            print("Rate limit exceeded, retrying after a delay...")
            time.sleep(60)
            return process_request_with_retry(client, model, prompt, max_tokens)
        return None


def transform(args) -> None:
    """Load data, summarize each chain-of-thought, and save the results to a JSON file."""
    data = load_json(args.input_json)
    prompts = [
        make_prompt(item.get("output", "").strip())
        for item in data
    ]

    # Initialize Azure OpenAI API client
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.api_key,
        api_version="2024-10-21"
    )

    results: List[Dict[str, Any]] = []

    # Process with a thread pool and show progress
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(
                process_request_with_retry, client, args.model, prompt, args.max_tokens
            ): idx
            for idx, prompt in enumerate(prompts)
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing",
            unit="item"
        ):
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
    parser = argparse.ArgumentParser(
        description="Summarize CoT outputs into concise responses"
    )
    parser.add_argument(
        "--input_json", required=True,
        help="Input JSON file containing 'instruction', 'input', and 'output' fields"
    )
    parser.add_argument(
        "--results_file", default="results.json",
        help="Output JSON file for the generated summaries"
    )
    parser.add_argument(
        "--model", required=True,
        help="Name of the OpenAI model to use"
    )
    parser.add_argument(
        "--azure_endpoint", required=True,
        help="Azure OpenAI API endpoint"
    )
    parser.add_argument(
        "--api_key", required=True,
        help="Azure OpenAI API key"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1000,
        help="Maximum number of tokens to generate per summary"
    )
    args = parser.parse_args()
    transform(args)
