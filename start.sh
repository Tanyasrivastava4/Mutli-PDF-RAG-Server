#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status

# Print failing command before exiting
trap 'echo "[ERROR] Command failed: $BASH_COMMAND"' ERR

echo "===== [Step 1] Cloning repository ====="
git clone https://github.com/Tanyasrivastava4/Mutli-PDF-RAG-Server.git
cd Mutli-PDF-RAG-Server

echo "===== [Step 2] Updating apt-get and installing system dependencies ====="
apt-get update && apt-get install -y libgl1 git

echo "===== [Step 3] Upgrading pip ====="
pip install --upgrade pip

echo "===== [Step 4] Installing Python dependencies ====="
pip install pdf2image pdfminer.six unstructured-inference pi_heif sentence-transformers
pip install -r requirements.txt

echo "===== [Step 5] Starting Uvicorn server ====="
uvicorn server_app:app --host 0.0.0.0 --port 8000
