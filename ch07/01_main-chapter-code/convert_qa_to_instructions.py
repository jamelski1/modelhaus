import json

# Input: your QA jsonl file
INPUT_PATH = "qa_dataset.jsonl"   # change if your file has a different name
OUTPUT_PATH = "qa_dataset.json"

SYSTEM_INSTRUCTION = (
    "Answer the following question about JP 3-12 cyberspace operations "
    "as clearly and concisely as possible."
)

def main():
    data = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            q = obj["question"].strip()
            a = obj["answer"].strip()

            entry = {
                "instruction": SYSTEM_INSTRUCTION,
                "input": q,
                "output": a,
            }
            data.append(entry)

    print(f"Converted {len(data)} QA pairs.")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
