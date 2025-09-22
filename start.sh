#!/bin/bash


# Make script executable and set error handling
chmod +x start.sh
set -e  # Exit immediately if a command fails
set -x  # Print commands as they execute
trap 'echo "[ERROR] Command failed: $BASH_COMMAND"' ERR


echo "===== [Step 0] Update system packages ====="
apt-get update -y
apt-get install -y git curl wget libgl1 python3-venv python3-pip netcat-traditional

echo "===== [Step 1] Clone repository if not exists ====="
if [ ! -d "Mutli-PDF-RAG-Server" ]; then
    git clone https://github.com/Tanyasrivastava4/Mutli-PDF-RAG-Server.git
else
    echo "[INFO] Repository already exists, updating..."
    cd Mutli-PDF-RAG-Server
    git pull origin main || echo "[WARN] Git pull failed, continuing with existing code"
    cd ..
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

# Core dependencies with specific versions
pip install fastapi==0.104.1 uvicorn==0.24.0

# ML dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.2 tokenizers==0.19.1 sentence-transformers==2.2.2

# Vector database
pip install chromadb

# PDF processing dependencies
pip install PyMuPDF==1.23.0 pdf2image==1.17.0 pdfminer.six==20221105

# Unstructured dependencies
pip install unstructured[pdf] unstructured-inference

# Additional utilities
pip install python-multipart requests pillow

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
fi

echo "===== [Step 4] Create necessary directories ====="
mkdir -p ./temp ./models

echo "===== [Step 5] Start health check in background ====="
# Start a simple health check server first
python3 -c "
import http.server
import socketserver
import threading
import time

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"status\": \"starting\"}')
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass  # Suppress default logging

def start_temp_server():
    with socketserver.TCPServer(('', 8000), HealthHandler) as httpd:
        httpd.timeout = 1
        while True:
            httpd.handle_request()

threading.Thread(target=start_temp_server, daemon=True).start()
time.sleep(2)
print('Temporary health server started on port 8000')
" &

TEMP_PID=$!
sleep 3

echo "===== [Step 6] Start Uvicorn server ====="
# Kill temporary server and start real server
kill $TEMP_PID 2>/dev/null || true
sleep 2

# Start the FastAPI server
echo "Starting FastAPI server..."
exec venv/bin/uvicorn server_app:app --host 0.0.0.0 --port 80 --workers 1 --timeout-keep-alive 300

