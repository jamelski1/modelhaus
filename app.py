#!/usr/bin/env python3
"""
HausGPT Web Interface
A Flask web application to interact with the fine-tuned JP 3-12 model.
"""

from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import os

app = Flask(__name__)

# Configuration
MODEL_ID = "jamelski/HausGPT"
MAX_LENGTH = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the model and tokenizer (called once at startup)."""
    global model, tokenizer

    print("Loading model and tokenizer...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
        model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def generate_response(prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P):
    """
    Generate a response from the model.

    Args:
        prompt: Input text
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter

    Returns:
        Generated text
    """
    if model is None or tokenizer is None:
        return "Error: Model not loaded"

    try:
        # Encode the prompt
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
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
