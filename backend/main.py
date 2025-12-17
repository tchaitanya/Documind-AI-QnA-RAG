import os
import time
import uuid
import asyncio
import tempfile
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient

# Document utilities
from document_utils import AzureBlobDocumentManager, DocumentProcessor, create_blob_manager, create_document_processor

# LangChain + Azure OpenAI + Azure AI Search
from langchain_community.vectorstores import AzureSearch as AzureSearchVS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI(title="Agentic RAG Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:5173")],
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

# Embeddings + LLM
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=EMBED_DEPLOYMENT,
        api_version=api_version,
    )
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=CHAT_DEPLOYMENT,
        api_version=api_version,
    )
else:
    print("Warning: Azure OpenAI not configured. Chat functionality will not work until you configure .env file.")
    embeddings = None
    llm = None

# Pre-initialize vectorstore at startup for better performance
_vectorstore_cache = None

def get_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is None:
        _vectorstore_cache = AzureSearchVS(
            azure_search_endpoint=SEARCH_SERVICE,
            azure_search_key=SEARCH_API_KEY,
            index_name=SEARCH_INDEX,
            embedding_function=embeddings.embed_query,
        )
        print(f"✅ Vectorstore initialized: {SEARCH_INDEX}")
    return _vectorstore_cache

# Initialize at startup
if embeddings is not None:
    get_vectorstore()
    print("✅ RAG system ready")

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.get("/health")
def health():
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
    try:
        # Process file using document processor
        print(f"[PROCESS] Loading and chunking {blob}...")
        chunks, chunk_count = doc_processor.process_file(blob_manager, blob)
        print(f"[PROCESS] Created {chunk_count} chunks")

        # Index into Azure AI Search
        print(f"[PROCESS] Indexing into Azure AI Search (hybrid mode)...")
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        print(f"[PROCESS] Successfully indexed {chunk_count} chunks")

        return {
            "blob": blob,
            "chunks": chunk_count,
            "chunks_indexed": chunk_count,
            "message": "Indexed into Azure AI Search"
        }
    
    except Exception as e:
        print(f"[PROCESS ERROR] Failed to process {blob}: {str(e)}")
        return {"error": f"Network Error: {str(e)}", "blob": blob, "chunks": 0}

@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    
    # Use cached vectorstore (already initialized at startup)
    vectorstore = get_vectorstore()
    
    # Hybrid search with fallback
    try:
        docs = vectorstore.similarity_search(req.query, k=req.top_k, search_type="hybrid")
    except:
        docs = vectorstore.similarity_search(req.query, k=req.top_k)
    search_time = time.time() - start

    # Build context (optimized)
    context = "\\n\\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}\\n{doc.page_content}" for doc in docs])
    
    # Generate answer (use global llm instance)
    llm_start = time.time()
    prompt_text = PROMPT_TEMPLATE.format(context=context, question=req.query)
    answer = llm.invoke(prompt_text).content
    llm_time = time.time() - llm_start
    
    # Prepare response with deduplicated sources (group by document name)
    sources_dict = {}
    for d in docs:
        source_name = d.metadata.get("source", "")
        if source_name not in sources_dict:
            sources_dict[source_name] = d.page_content
        else:
            # Append content from same source with separator
            sources_dict[source_name] += f"\n\n---\n\n{d.page_content}"
    
    sources = [{"source": name, "content": content} for name, content in sources_dict.items()]
    
    # Grounding check (optimized - tuple lookup)
    ungrounded_phrases = ("i don't have", "not in the", "no information", "cannot find", "not available", "not mentioned", "not provided")
    is_grounded = not any(p in answer.lower() for p in ungrounded_phrases)
    
    # Simplified reasoning log (pre-formatted strings)
    total_time = time.time() - start
    reasoning_log = [
        {"phase": "Retrieval", "details": f"Retrieved {len(docs)} documents using hybrid search", "duration": f"{search_time:.2f}s"},
        {"phase": "Generation", "details": f"Model: {CHAT_DEPLOYMENT} | Context size: {len(context)} chars", "duration": f"{llm_time:.2f}s"},
        {"phase": "Grounding", "details": f"Answer is {'grounded in documents' if is_grounded else 'missing information'}", "duration": "N/A"},
        {"phase": "Total", "details": f"End-to-end response time", "duration": f"{total_time:.2f}s"}
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "agentic": is_grounded,
        "reasoning_log": reasoning_log
    }

# Streaming endpoint (LLM-only streaming of final synthesis text)
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    print(f"\n{'='*80}")
    print(f"[STREAM REQUEST] Query: {req.query}")
    print(f"[STREAM REQUEST] Top K: {req.top_k}")
    print(f"{'='*80}")
    
    vectorstore = get_vectorstore()
    
    # Search documents (same as non-streaming)
    print(f"[STREAM] Searching for similar documents (k={req.top_k})...")
    print(f"[STREAM] Using hybrid search (vector + keyword)")
    docs = vectorstore.similarity_search(req.query, k=req.top_k, search_type="hybrid")
    print(f"[STREAM] Found {len(docs)} documents")

    # Load prompt template from file
    agentic_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Build context
    context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])
    prompt_text = agentic_prompt.format(context=context, question=req.query)
    
    print(f"[STREAM] Starting streaming response...")

    # Create streaming LLM
    streaming_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=CHAT_DEPLOYMENT,
        streaming=True,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )

    async def streamer() -> AsyncGenerator[bytes, None]:
        try:
            # Stream the response
            chunks_sent = 0
            async for chunk in streaming_llm.astream(prompt_text):
                if chunk.content:
                    yield chunk.content.encode("utf-8")
                    chunks_sent += 1
            
            print(f"[STREAM] Completed successfully ({chunks_sent} chunks sent)")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"[STREAM ERROR] {str(e)}")
            yield f"\n\nError: {str(e)}".encode("utf-8")

    return StreamingResponse(streamer(), media_type="text/plain")
