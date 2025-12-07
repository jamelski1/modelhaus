#!/usr/bin/env python3
"""
HausGPT Web Interface
A Flask web application to interact with the fine-tuned JP 3-12 model.
"""

from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download
from previous_chapters import GPTModel, generate as gpt_generate
import tiktoken
import torch
import json
import os

app = Flask(__name__)

# Configuration
MODEL_ID = "jamelski/HausGPT"  # Hugging Face repo with checkpoint files
MAX_LENGTH = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# Global variables for model and tokenizer
model = None
tokenizer = None
model_config = None


def load_model():
    """Load the custom GPTModel and tiktoken tokenizer from Hugging Face Hub."""
    global model, tokenizer, model_config

    print("Loading model and tokenizer from Hugging Face Hub...")
    try:
        # Download checkpoint and config files from Hub
        print(f"Downloading files from {MODEL_ID}...")

        checkpoint_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="gpt2-medium355M-sft.pth"
        )

        hparams_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="hparams.json"
        )

        print("Files downloaded successfully")

        # Load tiktoken tokenizer (standard GPT-2 encoding, matching training setup)
        print("Loading tiktoken tokenizer...")
        tokenizer = tiktoken.get_encoding("gpt2")

        # Load model config from hparams.json
        print("Loading model config...")
        with open(hparams_path, 'r') as f:
            hparams = json.load(f)

        print(f"Original config keys: {list(hparams.keys())}")
        print(f"Original config: {hparams}")

        # Map hparams.json keys to GPTModel expected keys
        cfg = {
            "vocab_size": hparams.get("n_vocab", 50257),
            "context_length": hparams.get("n_ctx", 1024),
            "emb_dim": hparams.get("n_embd", 1024),
            "n_heads": hparams.get("n_head", 16),
            "n_layers": hparams.get("n_layer", 24),
            "drop_rate": hparams.get("drop_rate", 0.1),  # Default dropout rate
            "qkv_bias": True  # Model was trained with bias in attention layers
        }

        print(f"Mapped config for GPTModel: {cfg}")

        # Store config globally for use in generate_response
        model_config = cfg

        # Create custom GPTModel instance
        print("Creating GPTModel...")
        model = GPTModel(cfg)

        # Load checkpoint weights
        print("Loading checkpoint weights...")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)

        # Move to GPU if available, use half precision to save memory
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Convert to half precision (float16) to reduce memory usage by ~50%
        print("Converting model to half precision (float16) to save memory...")
        model.half()

        model.eval()

        print(f"Model loaded successfully on {device}")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_response(prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P):
    """
    Generate a response from the model.

    Args:
        prompt: Input text
        max_length: Maximum length of generated text (total, including prompt)
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter (not used with current generate function)

    Returns:
        Generated text
    """
    global model, tokenizer, model_config

    if model is None or tokenizer is None:
        return "Error: Model not loaded"

    try:
        # Encode the prompt using tiktoken
        device = next(model.parameters()).device
        token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})

        # Convert to tensor and add batch dimension
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Calculate how many new tokens to generate
        max_new_tokens = max_length - len(token_ids)
        if max_new_tokens <= 0:
            max_new_tokens = 50  # Generate at least 50 tokens

        # Generate using the custom generate function from previous_chapters
        # Note: top_p is not supported by the generate function, using top_k=50 instead
        # Using positional arguments for compatibility
        output_ids = gpt_generate(
            model,
            input_ids,
            max_new_tokens,
            model_config["context_length"],
            temperature,
            50,  # top_k as approximation for top_p
            None  # eos_id - let it generate fully
        )

        # Decode the response using tiktoken
        output_tokens = output_ids[0].tolist()
        response = tokenizer.decode(output_tokens)

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"


@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint to generate text."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_length = int(data.get('max_length', MAX_LENGTH))
        temperature = float(data.get('temperature', TEMPERATURE))
        top_p = float(data.get('top_p', TOP_P))

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Generate response
        response = generate_response(prompt, max_length, temperature, top_p)

        return jsonify({
            'prompt': prompt,
            'response': response,
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and tokenizer is not None
    })


# Load model when the module is imported (for Gunicorn)
print("Initializing application...")
load_model()


if __name__ == '__main__':
    # Get port from environment variable (Render uses PORT)
    port = int(os.environ.get('PORT', 5000))

    # Run the app (model is already loaded above)
    app.run(host='0.0.0.0', port=port, debug=False)
