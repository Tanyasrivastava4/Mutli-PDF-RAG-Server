#!/bin/bash
set -e  # Exit immediately if a command fails
set -x  # Print commands as they execute

trap 'echo "[ERROR] Command failed: $BASH_COMMAND"' ERR

echo "===== [Step 0] Update system packages ====="
apt-get update -y
apt-get install -y git curl wget libgl1 python3-venv python3-pip

echo "===== [Step 1] Clone repository if not exists ====="
if [ ! -d "Mutli-PDF-RAG-Server" ]; then
    git clone https://github.com/Tanyasrivastava4/Mutli-PDF-RAG-Server.git
else
    echo "[INFO] Repository already exists, skipping clone"
fi

cd Mutli-PDF-RAG-Server

echo "===== [Step 2] Setup virtual environment inside repo ====="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "===== [Step 3] Upgrade pip and install dependencies ====="
pip install --upgrade pip

# Core dependencies
pip install fastapi uvicorn torch transformers sentence-transformers chromadb

# PDF + unstructured dependencies
pip install pdf2image pdfminer.six unstructured unstructured-inference pi_heif PyMuPDF

echo "===== [Step 0.5] Install Poppler for PDF processing ====="
apt-get install -y poppler-utils

echo "===== [Step 3.1] Install Python dependencies for local UI ====="
pip install requests jinja2


# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "===== [Step 4] Check available memory ====="
python3 - <<EOF
import psutil
available_gb = psutil.virtual_memory().available / 1e9
print(f"[MEMORY CHECK] Available RAM: {available_gb:.2f} GB")
if available_gb < 10:
    print("[WARNING] Available memory is low. Loading Mistral-7B may fail.")
EOF

echo "===== [Step 5] Start Uvicorn server ====="
# Replace shell with uvicorn process to keep container alive

exec venv/bin/uvicorn server_app:app \
    --host :: \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 300


    
