#!/bin/bash
chmod +x start.sh
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

# Activate venv
source venv/bin/activate

echo "===== [Step 3] Upgrade pip and install dependencies ====="
pip install --upgrade pip

# Core dependencies
pip install fastapi uvicorn torch transformers sentence-transformers chromadb

# PDF + unstructured dependencies
pip install pdf2image pdfminer.six unstructured unstructured-inference pi_heif

# Fix for error: fitz module missing (PyMuPDF provides it)
pip install PyMuPDF

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "===== [Step 4] Start Uvicorn server ====="
venv/bin/uvicorn server_app:app --host 0.0.0.0 --port 80
