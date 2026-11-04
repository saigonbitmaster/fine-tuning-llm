Markdown
# Fine-Tune & Deploy Custom Model with Ollama

A self-contained, end-to-end lab demonstrating how to **fine-tune a lightweight LLM using LoRA (CPU-friendly) $\rightarrow$ merge weights $\rightarrow$ export to GGUF $\rightarrow$ register and serve locally with Ollama (Snap)**.

---

## 🎯 Why This Lab Is Needed

In real-world software engineering and enterprise AI application development, off-the-shelf models often fall short for specific use cases. Creating a custom-tuned model is essential when you need:

- **Strict Domain Alignment:** Enforcing exact system identity, domain knowledge, proprietary terminology, or internal architecture context.
- **Data Privacy & Air-Gapped Compliance:** Keeping internal data and domain logic entirely local without relying on external cloud APIs.
- **Predictable Output Structuring:** Training models to consistently output targeted responses, custom formats, or tailored tool-use schemas.
- **Cost & Hardware Efficiency:** Fine-tuning a compact 0.5B/3B model to handle specialized tasks locally on standard CPU infrastructure instead of burning tokens on expensive cloud endpoints.

This lab provides a reproducible blueprint to build, export, and run your own domain-adapted model locally in minutes.

---

## 💻 Hardware Requirements & Performance

This pipeline is optimized for minimal hardware footprints and runs **extremely fast even on modest consumer hardware**:

- **Tested Configuration:** Intel Core i7 (4 Cores / 8 Threads) & 16 GB RAM.
- **No Discrete GPU Required:** Entire SFT training, weight merging, GGUF quantizing, and Ollama inference execute natively on standard consumer CPUs.
- **Resource Footprint:** Consumes less than 4 GB RAM during fine-tuning and inference.

---

## 🛠️ Prerequisites

- **OS:** Linux (Ubuntu 22.04 / 24.04 recommended)
- **Dependencies:** `git`, `python3`, `python3-venv`, `ollama` (installed via Snap)

Install required system packages:
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-full git
📂 Project Structure
Plaintext
fine-tune-ollama-lab/
├── README.md           # Lab documentation and rationale
├── requirements.txt    # Python dependencies
├── dataset.jsonl       # Custom SFT dataset (ChatML format)
├── train.py            # Training, weight merging, and GGUF export script
├── Modelfile.template  # Template for Ollama model configuration
└── run_pipeline.sh     # Automated end-to-end setup script
🚀 Quick Start
Make the pipeline script executable and run the full lab from scratch:

Bash
chmod +x run_pipeline.sh
./run_pipeline.sh
💡 Key Architectural Insights
Snap Sandbox Restrictions: Ollama installed via Snap runs in a strict sandbox environment. The generated .gguf file must be located inside the home directory (~/) for Ollama to access it properly.

Mandatory GGUF Export: Direct Hugging Face safetensors directory imports can trigger architecture mismatch errors (unsupported MLX architecture). Converting the merged weights to .gguf using llama.cpp ensures seamless compatibility with Ollama.

Stop Token Configuration: For Qwen2.5 architectures, PARAMETER stop "<|im_end|>" must be defined in the Modelfile to prevent truncated or runaway generations.