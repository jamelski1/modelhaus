#!/usr/bin/env python3
"""
Build a QA dataset from JP 3-12 PDF for fine-tuning.

This script:
  1. Extracts text from a PDF
  2. Chunks it into "contexts"
  3. Generates question–answer pairs for each context
     - Either as placeholders (mode: placeholder)
     - Or via Claude (mode: llm, using Claude Sonnet 4.5)
  4. Saves the dataset in JSON and JSONL formats, with schema:

     {
       "instruction": "Answer the question based on the context from JP 3-12.",
       "input": "Context:\\n...\\n\\nQuestion:\\n...",
       "output": "Short factual answer..."
     }
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# PDF → context extraction
# ---------------------------------------------------------------------------

def extract_contexts(pdf_path: str,
                     min_chars: int = 200,
                     max_chars: int = 2000) -> List[str]:
    """
    Extract text from PDF and chunk into contexts.

    Args:
        pdf_path: Path to the PDF file
        min_chars: Minimum characters per chunk
        max_chars: Maximum characters per chunk

    Returns:
        List of context strings suitable for QA generation
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("ERROR: pypdf is not installed. Run: pip install pypdf")
        sys.exit(1)

    print(f"Reading PDF: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"Total pages: {total_pages}")
    except Exception as e:
        print(f"ERROR: Failed to read PDF: {e}")
        sys.exit(1)

    # Extract text from all pages
    all_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"  WARNING: Failed to extract text from page {page_num}: {e}")
            continue

        if text and text.strip():
            all_text.append(text)

        if page_num % 10 == 0 or page_num == total_pages:
            print(f"  Extracted text from {page_num}/{total_pages} pages.")

    print(f"Extracted text from {len(all_text)} non-empty pages")

    # Combine all text
    combined_text = "\n\n".join(all_text)

    # Split into chunks on blank lines
    raw_chunks = re.split(r"\n\s*\n", combined_text)

    contexts: List[str] = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        chunk = re.sub(r"\s+", " ", chunk)  # normalize whitespace

        if min_chars <= len(chunk) <= max_chars:
            contexts.append(chunk)

    print(f"Created {len(contexts)} contexts (filtered {min_chars}-{max_chars} chars)")
    return contexts


# ---------------------------------------------------------------------------
# Placeholder QA (for testing)
# ---------------------------------------------------------------------------

def generate_placeholder_qa(context: str) -> List[Dict[str, str]]:
    """
    Generate placeholder QA pairs for testing the pipeline.
    """
    return [
        {
            "question": "What is this section of JP 3-12 mainly about?",
            "answer": "TODO: fill in manually."
        }
    ]


# ---------------------------------------------------------------------------
# LLM-based QA using Claude Sonnet 4.5
# ---------------------------------------------------------------------------

def generate_llm_qa(context: str, api_key: str) -> List[Dict[str, str]]:
    """
    Generate QA pairs using Claude AI via the Anthropic API (Claude Sonnet 4.5).

    Args:
        context: The context text
        api_key: Anthropic API key

    Returns:
        List of QA dictionaries with 'question' and 'answer' keys
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: anthropic package is not installed. Run: pip install anthropic")
        sys.exit(1)

    # Number of QA pairs to generate per context (tweak if you want more)
    n_questions = 3

    PROMPT_TEMPLATE = """You are creating high-quality question–answer pairs for a training dataset about
Joint Publication (JP) 3-12, Cyberspace Operations.

Your ONLY source of truth is the CONTEXT provided. You must not invent facts.

GOAL
- Produce {n_questions} doctrinally useful question–answer pairs.
- Each pair should help someone learn or test their understanding of JP 3-12.

STRICT OUTPUT FORMAT
- Respond with ONLY a JSON array.
- Each element must be an object with exactly two string fields: "question" and "answer".
- NO additional fields, NO explanations, NO prose, NO code fences.
- The output MUST be valid JSON.

VALID FORMAT EXAMPLE (structure only):

[
  {{"question": "According to JP 3-12, what is cyberspace?", "answer": "Cyberspace is ..."}},
  {{"question": "Which organization is responsible for X?", "answer": "X is the responsibility of ..."}}
]

CONTENT RULES

1. CONTEXT-BASED ONLY
   - Every question MUST be answerable directly and solely from the provided CONTEXT.
   - Never use outside knowledge of JP 3-12.
   - If the CONTEXT lacks detail, focus on what *is* stated.

2. FOCUS ON DOCTRINE (NOT TRIVIA)
   Prefer questions about:
   - purposes and objectives
   - roles, authorities, and responsibilities
   - command relationships
   - conditions and criteria
   - processes, interactions, and key relationships
   - definitions of doctrinal terms or program names

   Avoid:
   - URLs, office locations, administrative numbers
   - trivial restatement of a single sentence

3. VARIETY AND NON-DUPLICATION
   - Each question MUST target a different idea or detail from the CONTEXT.
   - Use diverse question types, such as:
       * "What is the purpose of X?"
       * "Under what conditions does Y occur?"
       * "Which organization is responsible for Z?"
       * "How does A support B?"
       * "What relationship exists between A and B?"

4. ANSWER STYLE
   - Answers must be concise, factual, and ideally 5–40 words.
   - Paraphrase when appropriate, but preserve doctrinal terminology.
   - Do NOT reference the CONTEXT or JP 3-12 explicitly (avoid phrases like "according to the context").
   - Just state the fact.

Now, using ONLY the information found in the CONTEXT below,
produce {n_questions} question–answer pairs in the required JSON format.

CONTEXT:
\"\"\"{context}\"\"\""""

    client = Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        n_questions=n_questions,
                        context=context
                    ),
                }
            ],
        )

        raw = message.content[0].text
        original_response = raw  # for debugging
        raw = raw.strip()

        # Strip Markdown code fences if present
        if raw.startswith("```"):
            # remove opening ``` or ```json
            raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        if raw.endswith("```"):
            raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        # Try to isolate just the JSON array (from first '[' to last ']')
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            raw_json = raw[start:end + 1]
        else:
            raw_json = raw

        qa_pairs = json.loads(raw_json)

        # If model returns a single object, wrap it in a list
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]

        if not isinstance(qa_pairs, list):
            print("  WARNING: LLM response is not a list/dict; skipping context")
            print(f"  Response preview: {original_response[:200].replace(os.linesep, ' ')}")
            return []

        valid_pairs: List[Dict[str, str]] = []
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            q = pair.get("question")
            a = pair.get("answer")
            if isinstance(q, str) and isinstance(a, str):
                q = q.strip()
                a = a.strip()
                if q and a:
                    valid_pairs.append({"question": q, "answer": a})
            else:
                print("  WARNING: Skipping invalid QA pair (missing 'question' or 'answer')")

        return valid_pairs

    except json.JSONDecodeError as e:
        print(f"  WARNING: Failed to parse LLM response as JSON: {e}")
        try:
            print(f"  Response preview: {original_response[:300].replace(os.linesep, ' ')}")
        except Exception:
            pass
        return []
    except Exception as e:
        print(f"  WARNING: LLM API call failed: {type(e).__name__}: {e}")
        return []


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(contexts: List[str],
                  mode: str = "placeholder",
                  api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Build the complete QA dataset from contexts.

    Args:
        contexts: List of context strings
        mode: "placeholder" or "llm"
        api_key: Anthropic API key (required if mode is "llm")

    Returns:
        List of dataset examples with instruction, input, output fields
    """
    dataset: List[Dict[str, str]] = []
    total_contexts = len(contexts)

    print(f"\nGenerating QA pairs in '{mode}' mode.")

    for idx, context in enumerate(contexts, start=1):
        if mode == "placeholder":
            qa_pairs = generate_placeholder_qa(context)
        elif mode == "llm":
            if not api_key:
                print("ERROR: API key is required for LLM mode")
                sys.exit(1)
            qa_pairs = generate_llm_qa(context, api_key)
        else:
            print(f"ERROR: Unknown mode '{mode}'. Use 'placeholder' or 'llm'.")
            sys.exit(1)

        for qa in qa_pairs:
            example = {
                "instruction": "Answer the question based on the context from JP 3-12.",
                "input": f"Context:\n{context}\n\nQuestion:\n{qa['question']}",
                "output": qa["answer"],
            }
            dataset.append(example)

        if idx % 10 == 0 or idx == total_contexts:
            print(f"  Processed {idx}/{total_contexts} contexts "
                  f"(generated {len(dataset)} examples so far)")

    print(f"\nTotal examples generated: {len(dataset)}")
    return dataset


# ---------------------------------------------------------------------------
# Save dataset
# ---------------------------------------------------------------------------

def save_dataset(dataset: List[Dict[str, str]],
                 output_prefix: str = "jp312_qa_dataset") -> None:
    """
    Save dataset in both JSON and JSONL formats.
    """
    json_path = f"{output_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON dataset to: {json_path}")

    jsonl_path = f"{output_prefix}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Saved JSONL dataset to: {jsonl_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Build a QA dataset from JP 3-12 PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate placeholder QA pairs (for testing)
  python build_jp312_qa_dataset.py JP_3-12_Cyberspace_Operations.pdf --mode placeholder

  # Generate QA pairs using Claude AI (requires ANTHROPIC_API_KEY env var)
  export ANTHROPIC_API_KEY="your-api-key-here"
  python build_jp312_qa_dataset.py JP_3-12_Cyberspace_Operations.pdf --mode llm
        """,
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file (e.g. JP_3-12_Cyberspace_Operations.pdf)",
    )
    parser.add_argument(
        "--mode",
        choices=["placeholder", "llm"],
        default="placeholder",
        help="QA generation mode: 'placeholder' or 'llm' (default: placeholder)",
    )
    parser.add_argument(
        "--output",
        default="jp312_qa_dataset",
        help="Output file prefix (default: jp312_qa_dataset)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum characters per context chunk (default: 200)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per context chunk (default: 2000)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    api_key = None
    if args.mode == "llm":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is not set")
            print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
            sys.exit(1)

    print("=" * 60)
    print("JP 3-12 QA Dataset Builder")
    print("=" * 60)

    contexts = extract_contexts(args.pdf_path, args.min_chars, args.max_chars)
    if not contexts:
        print("ERROR: No valid contexts extracted from PDF")
        sys.exit(1)

    dataset = build_dataset(contexts, mode=args.mode, api_key=api_key)
    if not dataset:
        print("WARNING: No dataset examples were generated")
        sys.exit(1)

    save_dataset(dataset, args.output)

    print("=" * 60)
    print("✓ Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
