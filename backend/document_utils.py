"""Document utilities for Azure Blob Storage operations and document processing."""

import os
from azure.storage.blob import BlobServiceClient, ContainerClient
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple
from langchain_core.documents import Document


class AzureBlobDocumentManager:
    """Manage uploads, downloads, and listings against a single blob container."""
    
    def __init__(self, connection_string: str, container_name: str):
        """
        Initialize the blob document manager.
        
        Args:
            connection_string: Azure Blob Storage connection string
            container_name: Name of the container to use
        """
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        self.container_client = self.blob_service.get_container_client(container_name)
    
    def upload_file(self, file_name: str, file_content: bytes, overwrite: bool = True) -> dict:
        """
        Upload a file to Azure Blob Storage.
        
        Args:
            file_name: Name of the file
            file_content: File content as bytes
            overwrite: Whether to overwrite existing file
            
        Returns:
            Dictionary with file metadata
        """
        blob_client = self.container_client.get_blob_client(file_name)
        blob_client.upload_blob(file_content, overwrite=overwrite)
        
        return {
            "name": file_name,
            "url": blob_client.url,
            "size": len(file_content)
        }
    
    def list_files(self) -> List[dict]:
        """
        List all files in the blob container.
        
        Returns:
            List of file metadata dictionaries
        """
        blobs = self.container_client.list_blobs()
        return [
            {
                "name": blob.name,
                "size": blob.size,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None
            }
            for blob in blobs
        ]
    
    def download_file(self, file_name: str) -> bytes:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            file_name: Name of the file to download
            
        Returns:
            File content as bytes
        """
        blob_client = self.container_client.get_blob_client(file_name)
        download_stream = blob_client.download_blob()
        return download_stream.readall()
    
    def delete_file(self, file_name: str) -> bool:
        """
        Delete a file from Azure Blob Storage.
        
        Args:
            file_name: Name of the file to delete
            
        Returns:
            True if successful
        """
        blob_client = self.container_client.get_blob_client(file_name)
        blob_client.delete_blob()
        return True


class DocumentProcessor:
    """Processes documents for RAG indexing."""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_document(self, file_path: str, file_name: str) -> List[Document]:
        """
        Load a document based on file extension.
        
        Args:
            file_path: Path to the file
            file_name: Original file name (for extension detection)
            
        Returns:
            List of Document objects
        """
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_name.endswith((".txt", ".md")):
            # Simple text loading for .txt and .md files
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": file_name})]
        elif file_name.endswith(".docx"):
            # Use UnstructuredFileLoader only for .docx
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        return self.text_splitter.split_documents(documents)
    
    def process_file(
        self, 
        blob_manager: AzureBlobDocumentManager, 
        file_name: str, 
        temp_dir: str = "temp"
    ) -> Tuple[List[Document], int]:
        """
        Download, load, and chunk a document from blob storage.
        
        Args:
            blob_manager: Azure blob manager instance
            file_name: Name of the file to process
            temp_dir: Temporary directory for downloads
            
        Returns:
            Tuple of (chunked documents, number of chunks)
        """
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download file from blob storage
        file_content = blob_manager.download_file(file_name)
        
        # Save temporarily to local disk
        local_path = os.path.join(temp_dir, f"temp_{file_name}")
        with open(local_path, "wb") as f:
            f.write(file_content)
        
        try:
            # Load document
            documents = self.load_document(local_path, file_name)
            print(f"[DEBUG] Loaded {len(documents)} raw documents from {file_name}")
            
            # Add source metadata and ensure page number exists
            for i, doc in enumerate(documents):
                doc.metadata["source"] = file_name
                # Ensure page metadata exists (required by Azure AI Search index)
                if "page" not in doc.metadata:
                    doc.metadata["page"] = i
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            print(f"[DEBUG] Created {len(chunks)} chunks from {file_name}")
            
            return chunks, len(chunks)
        
        finally:
            # Cleanup temporary file
            if os.path.exists(local_path):
                os.remove(local_path)


def create_blob_manager(connection_string: str, container_name: str) -> AzureBlobDocumentManager:
    """
    Factory function to create a blob manager.
    
    Args:
        connection_string: Azure Blob Storage connection string
        container_name: Container name
        
    Returns:
        AzureBlobDocumentManager instance
    """
    return AzureBlobDocumentManager(connection_string, container_name)


def create_document_processor(chunk_size: int = 2000, chunk_overlap: int = 200) -> DocumentProcessor:
    """
    Factory function to create a document processor.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        DocumentProcessor instance
    """
    return DocumentProcessor(chunk_size, chunk_overlap)
