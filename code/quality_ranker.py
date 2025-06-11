# select_best_cot.py
"""Batch‑evaluate multiple Chain‑of‑Thought (CoT) analyses for each item and
keep only the two best‑scoring CoTs.

The workflow mirrors `batch_inference`/`iterative_process_intermediate_file`
from the previous script so that the *entire* job (prompting, scoring,
selection and aggregation) finishes in as few passes over the model as
possible and supports automatic retry on malformed LLM outputs.

Usage (example):

```bash
python quality_ranker.py \
  --input_json data/all_items.json \
  --model_path /path/to/verifier \
  --intermediate_file tmp/top2_intermediate.json \
  --final_output data/top2_cots.json
```
"""

import argparse
import json
import os
import re
import gc
from typing import Dict, List, Any

from vllm import LLM, SamplingParams

######################################################################
# Generic helpers (load/save/extract‑JSON) – identical to previous script
######################################################################

def load_json(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf‑8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf‑8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def extract_json(text: str) -> Dict | None:
    """Robust JSON extraction (same heuristics as earlier script)."""
    text = text.strip()
    # 1) direct try
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) greedy inner braces
    for match in re.findall(r"{.*?}", text, flags=re.S):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    # 3) single quotes & missing quotes for keys → liberal fix‑ups
    fixed = text.replace("'", '"')
    fixed = re.sub(r"(\w+)\s*:", r'"\1":', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None

######################################################################
# Prompt template for scoring a single CoT
######################################################################

def cot_score_prompt(question: str, options: list[str] | list, answer: str, cot: str) -> str:
    """Return an evaluation prompt that asks the model to *score* this CoT.

    The verifier must output a JSON like {"score": 4, "justification": "..."}
    where score is 0‑5 (5 = perfect reasoning).
    """
    opt_lines = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
    return f"""You are a medical reasoning evaluator. Assess the following chain‑of‑thought (CoT) for its soundness and usefulness in arriving at the correct answer.

[Question]
{question}

[Options]
{opt_lines}

[Correct Answer]
{answer}

[CoT]
{cot}

Give a JSON object with two keys:
  "score": integer 0‑5 (5 = flawless reasoning, 0 = irrelevant/incorrect),
  "justification": one concise sentence.
"""

######################################################################
# Main batch + iterative processing
######################################################################

def batch_score(args: argparse.Namespace, llm: LLM) -> None:
    """Create every scoring prompt and run them in *one* LLM call batch.\n
    Results are stored in an intermediate file exactly once, mirroring the
    first phase of the reference script.
    """
    data = load_json(args.input_json)
    tasks: list[dict] = []
    cot_key_re = re.compile(r"^model\d+_COT\d+$")

    for idx, item in enumerate(data):
        base = {k: v for k, v in item.items() if k not in ("correct", "error")}
        for key, cot_text in item.get("correct", {}).items():
            if cot_key_re.match(key):
                prompt = cot_score_prompt(
                    question=item.get("question", ""),
                    options=item.get("options", []),
                    answer=item.get("answer", ""),
                    cot=cot_text)
                tasks.append({
                    "item_index": idx,
                    "cot_key": key,
                    "cot_text": cot_text,
                    "base_info": base,
                    "prompt": prompt,
                })

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</json>"]
    )

    # Run the entire batch – this *might* be large; adjust batch_size if OOM
    outputs = llm.generate([t["prompt"] for t in tasks], sampling)
    for t, out in zip(tasks, outputs):
        t["llm_raw_response"] = out.outputs[0].text.strip()

    save_json(tasks, args.intermediate_file)


def iterative_fix_and_select(args: argparse.Namespace, llm: LLM) -> None:
    """Iteratively repair malformed JSON, score again up to N times, then pick top‑2."""
    tasks: list[dict] = load_json(args.intermediate_file)
    if not tasks:
        print("No tasks to process – run batch_score first.")
        return

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</json>"]
    )

    # 1️⃣ parse existing responses, mark "score" if valid
    for t in tasks:
        js = extract_json(t.get("llm_raw_response", ""))
        if js and "score" in js and isinstance(js["score"], int):
            t["score"] = max(0, min(5, js["score"]))
            t["justification"] = js.get("justification", "")

    # 2️⃣ repair loop – keep only tasks w/o score
    retry_tasks = [t for t in tasks if "score" not in t]
    iteration = 0
    while retry_tasks and iteration < args.max_iterations:
        print(f"Iteration {iteration+1}: repairing {len(retry_tasks)} malformed outputs …")
        outs = llm.generate([t["prompt"] for t in retry_tasks], sampling)
        for t, out in zip(retry_tasks, outs):
            t["llm_raw_response"] = out.outputs[0].text.strip()
            js = extract_json(t["llm_raw_response"])
            if js and "score" in js and isinstance(js["score"], int):
                t["score"] = max(0, min(5, js["score"]))
                t["justification"] = js.get("justification", "")
        retry_tasks = [t for t in tasks if "score" not in t]
        iteration += 1

    # 3️⃣ give up on still‑broken ones: score = 0
    for t in retry_tasks:
        t["score"] = 0
        t["justification"] = "Invalid JSON after max retries."

    ##################################################################
    # Select best‑two per item_index
    ##################################################################
    per_item: dict[int, list[dict]] = {}
    for t in tasks:
        per_item.setdefault(t["item_index"], []).append(t)

    final_records: List[dict] = []
    for idx, task_list in per_item.items():
        # Sort by score desc, then keep highest two
        best_two = sorted(task_list, key=lambda d: d["score"], reverse=True)[:2]
        base = best_two[0]["base_info"].copy()  # they share the same base
        # Flatten into required format
        for t in best_two:
            base[t["cot_key"]] = t["cot_text"]
        final_records.append(base)

    # Sorted order for deterministic output
    final_records.sort(key=lambda d: d.get("question", ""))
    save_json(final_records, args.final_output)

    # ✂️ wipe intermediate file because everything now handled
    save_json([], args.intermediate_file)


######################################################################
# CLI entry
######################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pick the best two CoT analyses per item.")
    p.add_argument("--input_json", required=True, help="Input file with full CoT data.")
    p.add_argument("--model_path", required=True, help="Verifier LLM path (vLLM compatible).")
    p.add_argument("--intermediate_file", default="top2_intermediate.json")
    p.add_argument("--final_output", default="top2_cots.json", help="Final cleaned result file.")

    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=80000)
    p.add_argument("--max_iterations", type=int, default=100)

    args = p.parse_args()

    verifier = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # Phase 1 – run everything once if no intermediate present or empty.
    if not os.path.exists(args.intermediate_file) or os.path.getsize(args.intermediate_file) == 0:
        batch_score(args, verifier)

    # Phase 2 – iterative repair + aggregation
    iterative_fix_and_select(args, verifier)

    del verifier
    gc.collect()
