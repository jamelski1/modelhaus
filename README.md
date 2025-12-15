# modelhaus
Question-Answer Fine-Tuned Cyber Model

This repository contains tools for building QA datasets from JP 3-12 and a web interface for the fine-tuned HausGPT model.

## HausGPT Web Interface

A Flask-based web application to interact with the fine-tuned JP 3-12 Cyberspace Operations model.

**Model**: [jamelski/HausGPT on Hugging Face](https://huggingface.co/jamelski/HausGPT)

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py

# Visit http://localhost:5000
```

### Features

- Clean, modern web interface
- Adjustable generation parameters (temperature, top_p, max_length)
- Example prompts for JP 3-12 questions
- Responsive design for mobile and desktop
- Fast inference with GPU support (if available)

---

## Training the Model

To train the HausGPT model yourself, use the training notebook:

```
ch07/01_main-chapter-code/ch07.ipynb
```

This Jupyter notebook contains the full training pipeline for fine-tuning the model.

---

## 📚 JP 3-12 QA Dataset Builder

A Python script to convert the JP 3-12 Cyberspace Operations PDF into a question-answer (QA) dataset for fine-tuning language models.

### Features

- Extracts text from PDF and chunks it into meaningful contexts
- Two generation modes:
  - **Placeholder mode**: Creates template QA pairs for testing
  - **LLM mode**: Uses Claude AI to automatically generate high-quality QA pairs
- Outputs in both JSON and JSONL formats
- Configurable chunk sizes and filtering
- Progress tracking and error handling

### Installation

#### 1. Install Python Dependencies

```bash
pip install pypdf anthropic
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

Or create a virtual environment (recommended):

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install pypdf anthropic
```

#### 2. Set Up Anthropic API Key (for LLM mode)

If you want to use the LLM mode to automatically generate QA pairs, you need an Anthropic API key.

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

To make it permanent, add it to your `~/.bashrc`, `~/.zshrc`, or system environment variables.

### Usage

#### Basic Usage - Placeholder Mode

Test the pipeline without API calls:

```bash
python build_jp312_qa_dataset.py JP_3-12_Cyberspace_Operations.pdf --mode placeholder
```

This will create:
- `jp312_qa_dataset.json` - Pretty-printed JSON array
- `jp312_qa_dataset.jsonl` - One JSON object per line

#### LLM Mode - AI-Generated QA Pairs

Generate high-quality QA pairs using Claude AI:

```bash
# Make sure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY='your-api-key-here'

# Run the script
python build_jp312_qa_dataset.py JP_3-12_Cyberspace_Operations.pdf --mode llm
```

#### Advanced Options

```bash
python build_jp312_qa_dataset.py JP_3-12_Cyberspace_Operations.pdf \
  --mode llm \
  --output my_custom_dataset \
  --min-chars 300 \
  --max-chars 1500
```

**Options:**
- `--mode`: Choose `placeholder` or `llm` (default: `placeholder`)
- `--output`: Output file prefix (default: `jp312_qa_dataset`)
- `--min-chars`: Minimum characters per context chunk (default: 200)
- `--max-chars`: Maximum characters per context chunk (default: 2000)

#### Help

```bash
python build_jp312_qa_dataset.py --help
```

### Output Format

Each example in the dataset follows this exact schema:

```json
{
  "instruction": "Answer the question based on the context from JP 3-12.",
  "input": "Context:\n<chunk of text from the PDF>\n\nQuestion:\n<question about that chunk>",
  "output": "<short answer that can be found in the context>"
}
```

#### Example Output

```json
{
  "instruction": "Answer the question based on the context from JP 3-12.",
  "input": "Context:\nCyberspace operations are the employment of cyberspace capabilities where the primary purpose is to achieve objectives in or through cyberspace. Such operations include computer network operations and activities to operate and defend the Global Information Grid.\n\nQuestion:\nWhat is the primary purpose of cyberspace operations?",
  "output": "The primary purpose of cyberspace operations is to achieve objectives in or through cyberspace."
}
```

### How It Works

1. **PDF Extraction**: Uses `pypdf` to extract text from all pages
2. **Chunking**: Splits text into paragraphs and filters by length (200-2000 chars by default)
3. **QA Generation**:
   - **Placeholder mode**: Creates template QA pairs for manual filling
   - **LLM mode**: Sends each context to Claude with a prompt asking for 1-2 QA pairs
4. **Dataset Building**: Transforms QA pairs into the exact schema required
5. **Output**: Saves to both JSON and JSONL files

### Customizing the LLM Prompt

The LLM prompt is defined as a Python string in the `generate_llm_qa()` function. You can edit it directly in `build_jp312_qa_dataset.py` to change:
- Number of QA pairs per context
- Question types
- Answer length
- Focus areas

Look for the `PROMPT_TEMPLATE` variable around line 110.

### Troubleshooting

**"pypdf is not installed"**
```bash
pip install pypdf
```

**"anthropic package is not installed"**
```bash
pip install anthropic
```

**"ANTHROPIC_API_KEY environment variable is not set"**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**"PDF file not found"**
- Make sure the PDF is in the current directory or provide the full path
- Check the filename spelling

### Dependencies

- **pypdf**: Lightweight PDF text extraction library
  - Why: Simple, pure Python, handles most PDFs well
  - Alternative: `pdfplumber` (heavier but better for complex layouts)

- **anthropic**: Official Anthropic Python client for Claude API
  - Required only for LLM mode
