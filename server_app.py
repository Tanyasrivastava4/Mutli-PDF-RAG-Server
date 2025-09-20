from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from utils.chunking import chunk_pdf
from utils.embeddings import get_embeddings
from utils.retrieval import store_chunks, query_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, asyncio

app = FastAPI(title="Multi-PDF RAG Server")

# Global vars
model = None
tokenizer = None
model_ready = False

@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-PDF RAG Server is running"}

@app.get("/health")
def health():
    if model_ready:
        return {"status": "ok"}
    else:
        return {"status": "loading"}   # readiness probe can still pass

# Background task to load the model
async def load_model():
    global model, tokenizer, model_ready
    model_name = "mistralai/Mistral-7B-v0.1"
    hf_token = os.environ.get("HF_TOKEN")
    cache_dir = "./models"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, cache_dir=cache_dir, device_map="auto"
    )
    model_ready = True
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_model())  # don’t block FastAPI startup

def generate_text(prompt: str, max_length: int = 200):
    if not model_ready:
        return "Model is still loading, try again later."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# PDF upload endpoint
@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...),
                      doc_type: str = Form("manual"),
                      author: str = Form("unknown"),
                      date: str = Form("unknown")):
    if not model_ready:
        return {"error": "Model is still loading, try again later."}

    file_path = f"./temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        chunks = chunk_pdf(file_path, doc_type=doc_type)
    except Exception as e:
        return {"error": str(e)}

    embeddings = get_embeddings(chunks)
    metadata_list = [{"doc_type": doc_type, "author": author, "date": date} for _ in chunks]

    store_chunks(chunks, metadata_list, embeddings, doc_type=doc_type)
    os.remove(file_path)

    return {"message": f"{len(chunks)} chunks processed and stored successfully!"}

# Q&A endpoint
class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask(data: Question):
    if not model_ready:
        return {"error": "Model is still loading, try again later."}

    top_chunks = query_chunks(data.question)
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {data.question}\nAnswer:"

    try:
        answer = generate_text(prompt)
    except Exception as e:
        return {"error": str(e)}

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_app:app", host="0.0.0.0", port=8888, reload=False)
