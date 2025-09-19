#!/bin/bash
chmod +x start.sh
set -e  # Exit immediately if a command fails
set -x  # Print commands as they execute

trap 'echo "[ERROR] Command failed: $BASH_COMMAND"' ERR

echo "===== [Step 0] Update system packages ====="
apt-get update -y
apt-get install -y git curl wget libgl1 python3-venv python3-pip  # added python3-pip explicitly

echo "===== [Step 1] Setup virtual environment ====="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "===== [Step 2] Upgrade pip ====="
pip install --upgrade pip
pip install fastapi uvicorn torch transformers sentence-transformers chromadb pdf2image pdfminer.six unstructured-inference pi_heif
pip install unstructured unstructured-inference


echo "===== [Step 3] Install essential Python packages ====="
# Explicitly install uvicorn first to make sure server can run
pip install uvicorn


# Install other required packages
REQUIRED_PACKAGES=(pdf2image pdfminer.six unstructured-inference pi_heif sentence-transformers)
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show $pkg &> /dev/null; then
        pip install $pkg
    fi
done

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "===== [Step 4] Clone repository if not exists ====="
if [ ! -d "Mutli-PDF-RAG-Server" ]; then
    git clone https://github.com/Tanyasrivastava4/Mutli-PDF-RAG-Server.git
else
    echo "[INFO] Repository already exists, skipping clone"
fi

cd Mutli-PDF-RAG-Server

echo "===== [Step 5] Start Uvicorn server ====="
uvicorn server_app:app --host 0.0.0.0 --port 8000

