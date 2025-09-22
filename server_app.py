from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from utils.chunking import chunk_pdf
from utils.embedding import get_embeddings
from utils.retrieval import store_chunks, query_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI(title="Multi-PDF RAG Server")

# -----------------------------
# Root & health endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-PDF RAG Server is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Hugging Face LLM safe load
# -----------------------------
model_name = "mistralai/Mistral-7B-v0.1"
hf_token = os.environ.get("HF_TOKEN")
cache_dir = "./models"

def load_model_safe():
    """
    Load tokenizer and model safely, handling low-memory environments.
    """
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
        print("Tokenizer loaded!")

        print("Loading model...")
        # Use device_map="auto" if GPU is available, else fallback to CPU
        device_map = "auto" if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, cache_dir=cache_dir, device_map=device_map
        )
        print("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print("[ERROR] Failed to load model:", e)
        raise e

tokenizer, model = load_model_safe()

# -----------------------------
# Text generation helper
# -----------------------------
def generate_text(prompt: str, max_length: int = 200):
    print("Step 1: Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Step 2: Generating output from model...")
    outputs = model.generate(**inputs, max_length=max_length)
    print("Step 3: Decoding output...")
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Step 4: Text generation complete")
    return decoded

# -----------------------------
# PDF upload endpoint
# -----------------------------
@app.post("/process-pdf/")
async def process_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form("manual"),
    author: str = Form("unknown"),
    date: str = Form("unknown")
):
    file_path = f"./temp_{file.filename}"
    print(f"Saving uploaded PDF: {file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        chunks = chunk_pdf(file_path, doc_type=doc_type)
        print(f"{len(chunks)} chunks created")
    except Exception as e:
        return {"error": f"Error during chunking: {str(e)}"}

    embeddings = get_embeddings(chunks)
    print("Embeddings generated")

    metadata_list = [{"doc_type": doc_type, "author": author, "date": date} for _ in chunks]
    store_chunks(chunks, metadata_list, embeddings, doc_type=doc_type)
    print("Chunks stored successfully")

    os.remove(file_path)
    print("Temporary file removed")

    return {"message": f"{len(chunks)} chunks processed and stored successfully!"}

# -----------------------------
# Question answering endpoint
# -----------------------------
class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask(data: Question):
    question_text = data.question.lower()
    print(f"Received question: {question_text}")

    top_chunks = query_chunks(data.question)
    print(f"Retrieved {len(top_chunks)} top chunks")

    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {data.question}\nAnswer:"

    try:
        answer = generate_text(prompt)
        print("Answer generated successfully")
    except Exception as e:
        return {"error": f"Error during text generation: {str(e)}"}

    return {"answer": answer}

# -----------------------------
# Run uvicorn if executed directly
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_app:app", host="0.0.0.0", port=8000, reload=False)
