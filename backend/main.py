import os
import time
import traceback
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Document utilities
from document_utils import create_blob_manager, create_document_processor
from rag_pipeline import RAGPipeline

load_dotenv()

app = FastAPI(title="Agentic RAG Backend")

# CORS - Allow both common frontend ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("CORS_ORIGIN", "http://localhost:5173"),
        "http://localhost:5174",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

BLOB_CONN = os.getenv("AZURE_BLOB_CONN_STRING")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "docs")

SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

# Load prompt template from file
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt_instructions.txt")
def load_prompt_template():
    """Load the system prompt from disk, falling back to a default template on error."""
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load prompt file, using default. Error: {e}")
        return """You are a helpful AI assistant. Answer based ONLY on the provided context.

Context from documents:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE = load_prompt_template()

# Initialize blob manager and document processor
if BLOB_CONN:
    blob_manager = create_blob_manager(BLOB_CONN, BLOB_CONTAINER)
    doc_processor = create_document_processor(chunk_size=2000, chunk_overlap=200)
    print(f"✅ Blob manager initialized: {BLOB_CONTAINER}")
    
    # Ensure container exists
    try:
        blob_manager.container_client.create_container()
    except Exception:
        pass  # Container already exists
else:
    print("Warning: Azure Blob Storage not configured. Upload functionality will not work.")
    blob_manager = None
    doc_processor = None

rag_pipeline = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    rag_pipeline = RAGPipeline(
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        chat_deployment=CHAT_DEPLOYMENT,
        embed_deployment=EMBED_DEPLOYMENT,
        search_endpoint=SEARCH_SERVICE,
        search_key=SEARCH_API_KEY,
        search_index=SEARCH_INDEX,
        prompt_template=PROMPT_TEMPLATE,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )
    rag_pipeline.get_vectorstore()
    print("✅ RAG system ready")
else:
    print("Warning: Azure OpenAI not configured. Chat functionality will not work until you configure .env file.")

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.get("/health")
def health():
    """Lightweight health probe for container orchestrators."""
    return {"status": "ok"}

@app.get("/files")
def list_files():
    """List all uploaded files in Blob Storage"""
    try:
        files = blob_manager.list_files()
        return {"files": files, "count": len(files)}
    except Exception as e:
        return {"error": str(e), "files": [], "count": 0}

@app.post("/upload")
async def upload_file(request: Request, files: List[UploadFile] = File(...)):
    """Upload one or more files to Blob Storage and return their metadata."""
    uploaded_files = []
    
    for file in files:
        data = await file.read()
        
        # Upload using blob manager
        file_info = blob_manager.upload_file(file.filename, data)
        uploaded_files.append({
            "name": file_info["name"],
            "location": "blob",
            "path": file_info["name"]
        })
    
    return {"files": uploaded_files, "message": f"Uploaded {len(uploaded_files)} file(s)"}

@app.post("/process")
def process_blob(request: Request, blob: str = Query(...)):
    """Download a blob, chunk it, and index the chunks into Azure AI Search."""
    try:
        print(f"\n[PROCESS] Starting processing for: {blob}")
        
        # Process file using document processor
        print(f"[PROCESS] Loading and chunking {blob}...")
        chunks, chunk_count = doc_processor.process_file(blob_manager, blob)
        print(f"[PROCESS] Created {chunk_count} chunks")
        
        if chunk_count == 0:
            print(f"[PROCESS WARNING] No chunks created for {blob}")
            return {
                "blob": blob,
                "chunks": 0,
                "chunks_indexed": 0,
                "message": "No chunks created - file may be empty or unsupported format"
            }

        # Index into Azure AI Search
        print(f"[PROCESS] Indexing {chunk_count} chunks into Azure AI Search...")
        vectorstore = rag_pipeline.get_vectorstore()
        vectorstore.add_documents(chunks)
        print(f"[PROCESS] ✓ Successfully indexed {chunk_count} chunks for {blob}\n")

        return {
            "blob": blob,
            "chunks": chunk_count,
            "chunks_indexed": chunk_count,
            "message": "Indexed into Azure AI Search"
        }
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[PROCESS ERROR] Failed to process {blob}")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{error_trace}")
        return {"error": str(e), "blob": blob, "chunks": 0}

@app.post("/chat")
def chat(req: ChatRequest):
    """Answer a question using RAG over the indexed documents."""
    return rag_pipeline.chat(req.query, req.top_k)

# Streaming endpoint (LLM-only streaming of final synthesis text)
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the synthesized answer tokens as they are generated."""
    async def streamer():
        async for chunk in rag_pipeline.stream_chat(req.query, req.top_k):
            yield chunk

    return StreamingResponse(streamer(), media_type="text/plain")
