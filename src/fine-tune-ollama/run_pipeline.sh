#!/bin/bash
set -e

echo "=== 1. Creating & Activating Virtual Environment ==="
# Tạo venv tên là .venv nếu chưa có
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Kích hoạt venv
source .venv/bin/activate

echo "=== 2. Installing Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== 3. Running Training & GGUF Conversion ==="
python3 train.py

echo "=== 4. Registering Model with Snap Ollama ==="
cp Modelfile.template ~/mindsync-gguf-Modelfile
ollama create mindsync-custom -f ~/mindsync-gguf-Modelfile

echo "=== 5. Testing the Fine-Tuned Model ==="
ollama run mindsync-custom "Who is the architect of MindSync?"