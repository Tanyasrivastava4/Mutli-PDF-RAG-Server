from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from utils.chunking import chunk_pdf
from utils.embeddings import get_embeddings
from utils.retrieval import store_chunks, query_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-PDF RAG Server")

# Global vars
model = None
tokenizer = None
model_ready = False
model_loading = False

@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-PDF RAG Server is running"}

@app.get("/health")
def health():
    """
    Always return 200 OK for Salad's readiness probe
    This allows traffic to reach the container even during model loading
    """
    return {
        "status": "ok", 
        "model_ready": model_ready,
        "model_loading": model_loading
    }

@app.get("/model-status")
def model_status():
    """Separate endpoint to check model loading status"""
    return {
        "model_ready": model_ready,
        "model_loading": model_loading,
        "message": "Model loaded successfully!" if model_ready else 
                   "Model is loading..." if model_loading else "Model loading not started"
    }

# Background task to load the model
async def load_model():
    global model, tokenizer, model_ready, model_loading
    try:
        model_loading = True
        logger.info("Starting model loading...")
        
        model_name = "mistralai/Mistral-7B-v0.1"
        hf_token = os.environ.get("HF_TOKEN")
        cache_dir = "./models"
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token, 
            cache_dir=cache_dir
        )
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=hf_token, 
            cache_dir=cache_dir, 
            device_map="auto",
            torch_dtype=torch.float16  # Use half precision to save memory
        )
        
        model_ready = True
        model_loading = False
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        model_loading = False
        model_ready = False

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server starting up...")
    # Start model loading in background (non-blocking)
    asyncio.create_task(load_model())
    logger.info("FastAPI server started, model loading in background...")

def generate_text(prompt: str, max_length: int = 200):
    if not model_ready:
        raise HTTPException(
            status_code=503, 
            detail="Model is still loading. Please check /model-status for updates."
        )
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():  # Save memory
            outputs = model.generate(**inputs, max_length=max_length, do_sample=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Text generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# PDF upload endpoint
@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...),
                      doc_type: str = Form("manual"),
                      author: str = Form("unknown"),
                      date: str = Form("unknown")):
    
    logger.info(f"Processing PDF: {file.filename}, type: {doc_type}")
    
    # Create temp directory if it doesn't exist
    os.makedirs("./temp", exist_ok=True)
    file_path = f"./temp/{file.filename}"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to {file_path}")
        
        # Process the PDF
        chunks = chunk_pdf(file_path, doc_type=doc_type)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Get embeddings
        embeddings = get_embeddings(chunks)
        logger.info("Generated embeddings")
        
        # Prepare metadata
        metadata_list = [{"doc_type": doc_type, "author": author, "date": date} for _ in chunks]
        
        # Store in vector database
        store_chunks(chunks, metadata_list, embeddings, doc_type=doc_type)
        logger.info("Stored chunks in vector database")
        
        return {
            "message": f"{len(chunks)} chunks processed and stored successfully!",
            "doc_type": doc_type,
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

# Q&A endpoint
class Question(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask/")
async def ask(data: Question):
    if not model_ready:
        return {
            "error": "Model is still loading. Please check /model-status for updates.",
            "model_status": "loading"
        }
    
    try:
        logger.info(f"Processing question: {data.question}")
        
        # Retrieve relevant chunks
        top_chunks = query_chunks(data.question, top_k=data.top_k)
        
        if not top_chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "chunks_found": 0
            }
        
        # Create context
        context = "\n".join(top_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {data.question}\nAnswer:"
        
        # Generate answer
        answer = generate_text(prompt)
        
        return {
            "answer": answer,
            "chunks_found": len(top_chunks),
            "question": data.question
        }
        
    except Exception as e:
        logger.error(f"Question processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

@app.get("/collections")
async def list_collections():
    """List available document collections and their counts"""
    try:
        from utils.retrieval import invoice_store, paper_store, combined_store
        
        collections = {
            "invoice_store": invoice_store.count(),
            "paper_store": paper_store.count(),
            "combined_store": combined_store.count()
        }
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_app:app", host="0.0.0.0", port=8000, reload=False)
