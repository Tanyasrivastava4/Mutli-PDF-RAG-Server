#!/bin/bash

# 1. Update system and install basic dependencies
apt-get update
apt-get install -y git libgl1

# 2. Upgrade pip and install Python packages
pip install --upgrade pip
pip install pdf2image pdfminer.six unstructured-inference pi_heif sentence-transformers

# 3. Install other Python requirements from requirements.txt
pip install -r requirements.txt



# 5. Start FastAPI server
uvicorn server_app:app --host 0.0.0.0 --port 8000

