import json
import time
import os
from typing import Dict, Any, List

# pip install anthropic
from anthropic import Anthropic, APIError, RateLimitError

# ========= CONFIG =========

INPUT_PATH = "modelhaus_dataset_v3.json"   # your current dataset
OUTPUT_PATH = "qa_dataset.jsonl"           # new Q/A file (jsonl is nice for training)
MODEL_NAME = "claude-opus-4-5-20251101"    # or whatever Claude model you prefer
MAX_ITEMS = None  # set to an int (e.g. 500) to debug on a subset first

# *** IMPORTANT ***
# For safety, DON'T hard-code your real key into this file.
# Instead, set it in your shell before running:
#   PowerShell:  $env:ANTHROPIC_API_KEY="YOUR_KEY_HERE"
#   CMD:         set ANTHROPIC_API_KEY=YOUR_KEY_HERE
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    raise SystemExit(
        "Please set ANTHROPIC_API_KEY in your environment before running this script."
    )

client = Anthropic(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are converting instruction-tuning triplets into simple question-answer pairs.\n"
    "You are given:\n"
    " - A short doctrinal or policy excerpt.\n"
    " - An instruction describing what to explain.\n"
    " - An existing answer.\n\n"
    "Your job:\n"
    "1. Write ONE clear question a student might ask about the excerpt.\n"
    "2. Write ONE concise, factual answer grounded ONLY in the excerpt and existing answer.\n"
    "3. Do NOT invent policy or doctrine details that are not implied in the text.\n"
    "4. Make the question answerable *only* from the excerpt (no outside knowledge).\n"
    "5. Return STRICTLY valid JSON with keys: 'question' and 'answer'."
)


def build_user_prompt(example: Dict[str, Any]) -> str:
    """
    Format a single training example into a prompt for Claude.
    The example is expected to have keys: instruction, input, output.
    """
    instruction = example.get("instruction", "").strip()
    excerpt = example.get("input", "").strip()
    answer = example.get("output", "").strip()

    return f"""
Here is a training example:

[EXCERPT]
{excerpt}

[INSTRUCTION]
{instruction}

[EXISTING ANSWER]
{answer}

TASK:
Rewrite this as a single QUESTION and a single ANSWER suitable for a Q&A style dataset.

Requirements:
- The QUESTION should be a natural language question that a learner could ask about the excerpt.
- The ANSWER should be a direct, factual response supported by the excerpt and consistent with the existing answer.
- Make both the question and answer clear but concise.
- Do not mention 'excerpt', 'instruction', or 'existing answer' in your output.
- Respond ONLY with a JSON object like:
  {{"question": "...", "answer": "..."}}
"""


def call_claude(prompt: str) -> Dict[str, str]:
    """
    Call Claude and parse a JSON {question, answer} object from the response.
    Includes basic retry and error handling.
    """
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=MODEL_NAME,
                max_tokens=256,
                temperature=0.1,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()

            # Sometimes models wrap JSON in code fences; strip them crudely
            if text.startswith("```"):
                text = text.strip("`")
                # remove leading language tag if present, e.g. json\n{...}
                if "\n" in text:
                    text = text.split("\n", 1)[1].strip()

            data = json.loads(text)
            if not isinstance(data, dict) or "question" not in data or "answer" not in data:
                raise ValueError("Response JSON missing 'question' or 'answer' keys")

            # Basic cleanup
            data["question"] = data["question"].strip()
            data["answer"] = data["answer"].strip()
            return data

        except (APIError, RateLimitError) as e:
            wait = 5 * (attempt + 1)
            print(f"[WARN] API error ({e}); retrying in {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Failed to parse Claude response on attempt {attempt+1}: {e}")
            print("Raw response text:")
            print("-" * 40)
            print(locals().get("text", "<no text captured>"))
            print("-" * 40)
            # On parse failure, re-raise so you notice and can fix the prompt
            raise

    raise RuntimeError("Failed to get a valid response from Claude after 3 attempts.")


def main():
    print(f"Loading dataset from {INPUT_PATH} ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if MAX_ITEMS is not None:
        data = data[:MAX_ITEMS]

    print(f"Total examples to process: {len(data)}")

    # If resuming, we don't want to overwrite previous results
    processed = 0
    mode = "w"
    with open(OUTPUT_PATH, mode, encoding="utf-8") as out_f:
        for idx, ex in enumerate(data, start=1):
            print(f"\n=== Example {idx}/{len(data)} ===")
            prompt = build_user_prompt(ex)
            qa = call_claude(prompt)
            json_line = json.dumps(qa, ensure_ascii=False)
            out_f.write(json_line + "\n")
            processed += 1
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}")
            # small delay to be nice to the API
            time.sleep(0.4)

    print(f"\nDone. Wrote {processed} Q/A pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
