#!/usr/bin/env python3
"""
Build a QA dataset from JP 3-12 PDF for fine-tuning.

This script extracts text from a PDF, chunks it into contexts,
and generates question-answer pairs either as placeholders or
using Claude AI via the Anthropic API.
"""

import json
import os
import re
import sys
from typing import List, Dict, Optional
import argparse


def extract_contexts(pdf_path: str, min_chars: int = 200, max_chars: int = 2000) -> List[str]:
    """
    Extract text from PDF and chunk into contexts.

    Args:
        pdf_path: Path to the PDF file
        min_chars: Minimum characters per chunk (default: 200)
        max_chars: Maximum characters per chunk (default: 2000)

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
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text and text.strip():
            all_text.append(text)
        if page_num % 10 == 0:
            print(f"  Extracted text from {page_num}/{total_pages} pages...")

    print(f"Extracted text from {len(all_text)} non-empty pages")

    # Combine all text
    combined_text = "\n\n".join(all_text)

    # Split into chunks (paragraphs)
    # Split on double newlines or multiple newlines
    raw_chunks = re.split(r'\n\s*\n', combined_text)

    # Filter and clean chunks
    contexts = []
    for chunk in raw_chunks:
        # Clean up whitespace
        chunk = chunk.strip()
        chunk = re.sub(r'\s+', ' ', chunk)  # Normalize whitespace

        # Filter by length
        if min_chars <= len(chunk) <= max_chars:
            contexts.append(chunk)

    print(f"Created {len(contexts)} contexts (filtered {min_chars}-{max_chars} chars)")

    return contexts


def generate_placeholder_qa(context: str) -> List[Dict[str, str]]:
    """
    Generate placeholder QA pairs for testing the pipeline.

    Args:
        context: The context text

    Returns:
        List of QA dictionaries with 'question' and 'answer' keys
    """
    return [
        {
            "question": "What is this section of JP 3-12 mainly about?",
            "answer": "TODO: fill in manually."
        }
    ]


def generate_llm_qa(context: str, api_key: str) -> List[Dict[str, str]]:
    """
    Generate QA pairs using Claude AI via the Anthropic API.

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

    # This is the LLM prompt - easily editable
    PROMPT_TEMPLATE = """You are helping create a question-answer dataset from a military doctrine document (JP 3-12 on Cyberspace Operations).

I will provide you with a context chunk from the document. Your task is to generate 1-2 high-quality question-answer pairs where:

1. Each question must be answerable ONLY from the given context
2. The answer must be concise and factual (1-3 sentences)
3. Questions should focus on key concepts, definitions, procedures, or important facts
4. Avoid yes/no questions - prefer "what", "how", "why", "who" questions
5. The answer should be a direct extract or close paraphrase from the context

Return your response as a JSON array of objects, where each object has exactly these two fields:
- "question": the question string
- "answer": the answer string

Example format:
[
  {
    "question": "What is the primary purpose of cyberspace operations?",
    "answer": "The primary purpose is to achieve military objectives in or through cyberspace."
  }
]

Context:
{context}

Generate 1-2 question-answer pairs in JSON format:"""

    client = Anthropic(api_key=api_key)

    try:
        # Call Claude API
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(context=context)
                }
            ]
        )

        # Extract the response text
        response_text = message.content[0].text

        # Try to parse as JSON
        # Sometimes the model wraps JSON in markdown code blocks
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        qa_pairs = json.loads(response_text)

        # Validate structure
        if not isinstance(qa_pairs, list):
            print(f"  WARNING: LLM response is not a list, skipping context")
            return []

        valid_pairs = []
        for pair in qa_pairs:
            if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                valid_pairs.append({
                    "question": pair["question"],
                    "answer": pair["answer"]
                })

        return valid_pairs

    except json.JSONDecodeError as e:
        print(f"  WARNING: Failed to parse LLM response as JSON: {e}")
        return []
    except Exception as e:
        print(f"  WARNING: LLM API call failed: {e}")
        return []


def build_dataset(contexts: List[str], mode: str = "placeholder", api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Build the complete QA dataset from contexts.

    Args:
        contexts: List of context strings
        mode: "placeholder" or "llm"
        api_key: Anthropic API key (required if mode is "llm")

    Returns:
        List of dataset examples with instruction, input, output fields
    """
    dataset = []
    total_contexts = len(contexts)

    print(f"\nGenerating QA pairs in '{mode}' mode...")

    for idx, context in enumerate(contexts, 1):
        # Generate QA pairs based on mode
        if mode == "placeholder":
            qa_pairs = generate_placeholder_qa(context)
        elif mode == "llm":
            if not api_key:
                print("ERROR: API key is required for LLM mode")
                sys.exit(1)
            qa_pairs = generate_llm_qa(context, api_key)
        else:
            print(f"ERROR: Unknown mode '{mode}'. Use 'placeholder' or 'llm'")
            sys.exit(1)

        # Transform each QA pair into the exact schema
        for qa in qa_pairs:
            example = {
                "instruction": "Answer the question based on the context from JP 3-12.",
                "input": f"Context:\n{context}\n\nQuestion:\n{qa['question']}",
                "output": qa["answer"]
            }
            dataset.append(example)

        # Progress update
        if idx % 10 == 0 or idx == total_contexts:
            print(f"  Processed {idx}/{total_contexts} contexts (generated {len(dataset)} examples so far)")

    print(f"\nTotal examples generated: {len(dataset)}")
    return dataset


def save_dataset(dataset: List[Dict[str, str]], output_prefix: str = "jp312_qa_dataset"):
    """
    Save dataset in both JSON and JSONL formats.

    Args:
        dataset: List of dataset examples
        output_prefix: Prefix for output files
    """
    # Save as JSON (pretty-printed array)
    json_path = f"{output_prefix}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON dataset to: {json_path}")

    # Save as JSONL (one object per line)
    jsonl_path = f"{output_prefix}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"Saved JSONL dataset to: {jsonl_path}")


def main():
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
        """
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file (e.g., JP_3-12_Cyberspace_Operations.pdf)"
    )
    parser.add_argument(
        "--mode",
        choices=["placeholder", "llm"],
        default="placeholder",
        help="QA generation mode: 'placeholder' for dummy data, 'llm' for AI-generated (default: placeholder)"
    )
    parser.add_argument(
        "--output",
        default="jp312_qa_dataset",
        help="Output file prefix (default: jp312_qa_dataset)"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum characters per context chunk (default: 200)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per context chunk (default: 2000)"
    )

    args = parser.parse_args()

    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Get API key if in LLM mode
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

    # Step 1: Extract contexts from PDF
    contexts = extract_contexts(args.pdf_path, args.min_chars, args.max_chars)

    if not contexts:
        print("ERROR: No valid contexts extracted from PDF")
        sys.exit(1)

    # Step 2: Build dataset
    dataset = build_dataset(contexts, mode=args.mode, api_key=api_key)

    if not dataset:
        print("WARNING: No dataset examples were generated")
        sys.exit(1)

    # Step 3: Save dataset
    save_dataset(dataset, args.output)

    print("=" * 60)
    print("✓ Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
