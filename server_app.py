from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from utils.chunking import chunk_pdf
from utils.embedding import get_embeddings
from utils.retrieval import store_chunks, query_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI(title="Multi-PDF RAG Server")

# Root endpoint
@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-PDF RAG Server is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Hugging Face LLM
model_name = "mistralai/Mistral-7B-v0.1"
hf_token = os.environ.get("HF_TOKEN")
cache_dir = "./models"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
print("Tokenizer loaded!")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, token=hf_token, cache_dir=cache_dir, device_map="auto"
)
print("Model loaded successfully!")

def generate_text(prompt: str, max_length: int = 200):
    print("Step 1: Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Step 2: Generating output from model...")
    outputs = model.generate(**inputs, max_length=max_length)
    print("Step 3: Decoding output...")
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Step 4: Text generation complete")
    return decoded

# PDF upload endpoint
@app.post("/process-pdf/")
async def process_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form("manual"),
    author: str = Form("unknown"),
    date: str = Form("unknown")
):
    print(f"Step 1: Saving uploaded PDF '{file.filename}'")
    file_path = f"./temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print("Step 2: Chunking PDF...")
    try:
        chunks = chunk_pdf(file_path, doc_type=doc_type)
        print(f"Step 3: {len(chunks)} chunks created")
    except Exception as e:
        print("Error during chunking:", e)
        return {"error": str(e)}

    print("Step 4: Generating embeddings for chunks...")
    embeddings = get_embeddings(chunks)
    print("Step 5: Embeddings generated")

    metadata_list = [{"doc_type": doc_type, "author": author, "date": date} for _ in chunks]

    print("Step 6: Storing chunks in database...")
    store_chunks(chunks, metadata_list, embeddings, doc_type=doc_type)
    print("Step 7: Chunks stored successfully")

    os.remove(file_path)
    print("Step 8: Temporary file removed")

    return {"message": f"{len(chunks)} chunks processed and stored successfully!"}

# Question answering endpoint
class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask(data: Question):
    print("Step 1: Received question")
    question_text = data.question.lower()
    print(f"Step 2: Question text processed: {question_text}")

    print("Step 3: Querying chunks for relevant context...")
    top_chunks = query_chunks(data.question)
    print(f"Step 4: Retrieved {len(top_chunks)} top chunks")

    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {data.question}\nAnswer:"

    print("Step 5: Generating answer using LLM...")
    try:
        answer = generate_text(prompt)
        print("Step 6: Answer generated successfully")
    except Exception as e:
        print("Error during text generation:", e)
        return {"error": str(e)}

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_app:app", host="0.0.0.0", port=8000, reload=False)
