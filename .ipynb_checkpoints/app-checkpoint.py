#!/usr/bin/env python3
"""
HausGPT Web Interface
Uses custom GPTModel + JP 3-12 QA dataset for context-aware answers.
"""

from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download
from previous_chapters import GPTModel, generate as gpt_generate
import tiktoken
import torch
import json
import os
import re

app = Flask(__name__)

# ------------------------- CONFIG -------------------------

MODEL_ID = "jamelski/HausGPT"
MAX_LENGTH = 200
TEMPERATURE = 0.7
TOP_P = 0.9  # approximated with top_k in generate()
DATASET_PATH = os.path.join(os.path.dirname(__file__), "jp312_qa_dataset.jsonl")

# Globals
model = None
tokenizer = None
model_config = None
qa_examples = []   # list of {"context", "question", "answer"}

# --------------------- DATASET LOADING --------------------

def load_qa_dataset():
    """Load JP 3-12 QA examples from JSONL for simple retrieval."""
    global qa_examples
    qa_examples = []

    if not os.path.exists(DATASET_PATH):
        print(f"[WARN] Dataset file not found: {DATASET_PATH}")
        return

    print(f"Loading QA dataset from {DATASET_PATH} ...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            inp = obj.get("input", "")
            out = obj.get("output", "").strip()

            # Expect pattern: "Context:\n...\n\nQuestion:\n..."
            ctx = inp
            q = ""
            if "Question:" in inp:
                ctx_part, q_part = inp.split("Question:", 1)
                ctx = ctx_part.replace("Context:", "").strip()
                q = q_part.strip()

            qa_examples.append({
                "context": ctx,
                "question": q,
                "answer": out,
            })

    print(f"Loaded {len(qa_examples)} QA examples.")

def _normalize(text: str):
    tokens = re.findall(r"\w+", text.lower())
    stop = {
        "the", "a", "an", "of", "in", "to", "and", "for", "on", "is", "are",
        "according", "jp", "3", "12", "joint", "publication", "doj", "dod"
    }
    return [t for t in tokens if t not in stop]

def _question_similarity(q1: str, q2: str) -> float:
    s1, s2 = set(_normalize(q1)), set(_normalize(q2))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def find_best_context(user_question: str):
    """Return best-matching context snippet from QA dataset."""
    if not qa_examples:
        return None

    best_ex = None
    best_sim = 0.0
    for ex in qa_examples:
        sim = _question_similarity(user_question, ex["question"])
        if sim > best_sim:
            best_sim = sim
            best_ex = ex

    # Basic similarity threshold to avoid nonsense matches
    if best_sim < 0.1:
        print(f"[INFO] No good match for question (best sim={best_sim:.3f})")
        return None

    print(f"[INFO] Best match sim={best_sim:.3f} ; question='{best_ex['question'][:80]}...'")
    return best_ex["context"]

# ----------------------- MODEL LOAD -----------------------

def load_model():
    """Load custom GPTModel and tiktoken tokenizer from Hugging Face Hub."""
    global model, tokenizer, model_config

    print("Loading model and tokenizer from Hugging Face Hub...")
    try:
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="gpt2-medium355M-sft.pth"
        )
        hparams_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="hparams.json"
        )

        print("Files downloaded successfully.")

        # Tokenizer
        tokenizer_local = tiktoken.get_encoding("gpt2")

        with open(hparams_path, "r") as f:
            hparams = json.load(f)

        print(f"Original config: {hparams}")

        cfg = {
            "vocab_size": hparams.get("n_vocab", 50257),
            "context_length": hparams.get("n_ctx", 1024),
            "emb_dim": hparams.get("n_embd", 1024),
            "n_heads": hparams.get("n_head", 16),
            "n_layers": hparams.get("n_layer", 24),
            "drop_rate": hparams.get("drop_rate", 0.1),
            "qkv_bias": True,
        }

        global tokenizer, model_config
        tokenizer = tokenizer_local
        model_config = cfg

        print("Creating GPTModel...")
        model_local = GPTModel(cfg)

        print("Loading checkpoint weights...")
        state = torch.load(checkpoint_path, map_location="cpu")
        model_local.load_state_dict(state)

        device = "cpu"
        model_local.to(device)
        model_local.eval()

        global model
        model = model_local

        print(f"Model loaded successfully on {device}.")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

# -------------------- PROMPT / GENERATION -----------------

def build_prompt(user_question: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a **direct answer**, not another question. "
        "Use clear, concise language.\n\n"
        "### Instruction:\n"
        f"{user_question.strip()}\n\n"
        "### Response:\n"
    )

def generate_response(user_prompt: str,
                      max_length: int = MAX_LENGTH,
                      temperature: float = TEMPERATURE,
                      top_p: float = TOP_P) -> str:
    """Generate a response from the model."""
    global model, tokenizer, model_config

    if model is None or tokenizer is None:
        return "Error: Model not loaded."

    try:
        full_prompt = build_prompt(user_prompt)

        device = next(model.parameters()).device
        prompt_token_ids = tokenizer.encode(
            full_prompt,
            allowed_special={"<|endoftext|>"}
        )
        input_ids = torch.tensor(
            prompt_token_ids, dtype=torch.long
        ).unsqueeze(0).to(device)

        max_new_tokens = max_length - len(prompt_token_ids)
        if max_new_tokens <= 0:
            max_new_tokens = 50

        output_ids = gpt_generate(
            model,
            input_ids,
            max_new_tokens,
            model_config["context_length"],
            temperature,
            50,   # top_k
            None  # eos_id
        )

        full_output_tokens = output_ids[0].tolist()
        completion_tokens = full_output_tokens[len(prompt_token_ids):]

        if not completion_tokens:
            return "No output generated."

        response = tokenizer.decode(completion_tokens).strip()

        # Cut if it starts generating the next example
        for stop_phrase in [
            "Below is an instruction",
            "### Instruction:",
            "### Input:",
            "### Response:"
        ]:
            idx = response.find(stop_phrase)
            if idx != -1:
                response = response[:idx].strip()
                break

        return response if response else "No output generated."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

# ------------------------- ROUTES -------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_prompt = data.get("prompt", "")
        max_length = int(data.get("max_length", MAX_LENGTH))
        temperature = float(data.get("temperature", TEMPERATURE))
        top_p = float(data.get("top_p", TOP_P))

        if not user_prompt:
            return jsonify({"error": "No prompt provided"}), 400

        response = generate_response(user_prompt, max_length, temperature, top_p)

        return jsonify({
            "prompt": user_prompt,
            "response": response,
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "qa_examples": len(qa_examples),
    })

# ------------------------ STARTUP ------------------------

print("Initializing application...")
load_qa_dataset()
load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
