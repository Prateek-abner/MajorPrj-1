# -*- coding: utf-8 -*-
"""
Data Processor - Loads PDFs, chunks, and vectorizes them
Run this FIRST to prepare your data
"""

import os
import sys
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch


# Configuration
PDF_FOLDER = "./data"
CHROMA_DB_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def discover_pdfs(folder: str) -> List[str]:
    """Auto-discover PDF files in the given folder."""
    if not os.path.exists(folder):
        print(f"⚠️ PDF folder not found: {folder}")
        print(f"📁 Creating folder: {folder}")
        os.makedirs(folder, exist_ok=True)
        return []
    
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if pdfs:
        print(f"📚 Found {len(pdfs)} PDF(s) in {folder}")
        for pdf in pdfs:
            print(f"   - {pdf}")
    else:
        print(f"⚠️ No PDF files found in {folder}")
    return pdfs


def load_pdfs(pdf_folder: str, files: List[str] = None) -> List:
    """Load PDFs from the specified folder."""
    all_docs = []
    
    if not os.path.exists(pdf_folder):
        print(f"⚠️ PDF folder not found: {pdf_folder}")
        return all_docs

    # Auto-discover PDFs if no specific files provided
    if files is None:
        files = discover_pdfs(pdf_folder)
        if not files:
            print(f"❌ No PDF files found in {pdf_folder}")
            print(f"📝 Please add PDF files to the '{pdf_folder}' folder and run again")
            return all_docs

    total_pages = 0
    print("\n📖 Loading PDFs...")
    for fname in files:
        path = os.path.join(pdf_folder, fname)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            total_pages += len(docs)
            print(f"✅ Loaded {len(docs)} page(s) from {fname}")
        except Exception as e:
            print(f"⚠️ Failed to load {fname}: {e}")
    
    if total_pages > 0:
        print(f"\n📄 Total pages loaded: {total_pages} from {len(files)} PDF(s)")
    return all_docs


def split_documents(documents: List, chunk_size: int = CHUNK_SIZE, 
                    chunk_overlap: int = CHUNK_OVERLAP):
    """Split documents into chunks for better retrieval."""
    print(f"\n✂️ Splitting documents into chunks...")
    print(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunk(s)")
    return chunks


def build_vectorstore(chunks: List, persist_directory: str = CHROMA_DB_DIR):
    """Build and persist a vector store from document chunks."""
    print(f"\n🔢 Building vector database...")
    
    # Configure embeddings with device placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32}
    )
    
    # Remove existing vectorstore if present
    if os.path.exists(persist_directory):
        import shutil
        print(f"🗑️ Removing existing vectorstore...")
        shutil.rmtree(persist_directory)
    
    # Create new vectorstore
    print(f"🆕 Creating new vectorstore at '{persist_directory}'...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vectordb.persist()
    print(f"💾 Vectorstore saved to disk")
    
    # Verify vectorstore
    collection_count = vectordb._collection.count()
    print(f"✅ Vectorstore contains {collection_count} embedding(s)")
    
    return vectordb


def process_documents(pdf_folder: str = PDF_FOLDER, 
                     persist_dir: str = CHROMA_DB_DIR,
                     force_reload: bool = True):
    """Main processing pipeline: Load PDFs -> Chunk -> Vectorize."""
    
    print("="*60)
    print("📊 DATA PROCESSOR - PDF to Vector Database")
    print("="*60)
    
    # Step 1: Load PDFs
    docs = load_pdfs(pdf_folder)
    if not docs:
        print("\n❌ ERROR: No documents loaded!")
        print(f"📝 Action required: Add PDF files to '{pdf_folder}' folder")
        sys.exit(1)
    
    # Step 2: Split into chunks
    chunks = split_documents(docs)
    if not chunks:
        print("\n❌ ERROR: No chunks created!")
        sys.exit(1)
    
    # Step 3: Build vectorstore
    vectordb = build_vectorstore(chunks, persist_directory=persist_dir)
    
    print("\n" + "="*60)
    print("✅ DATA PROCESSING COMPLETE!")
    print("="*60)
    print(f"📂 Vector database saved to: {persist_dir}")
    print(f"📊 Total documents processed: {len(docs)}")
    print(f"📦 Total chunks created: {len(chunks)}")
    print("\n🚀 You can now run chatbot.py or app.py")
    print("="*60)
    
    return vectordb


if __name__ == "__main__":
    # Run the data processing pipeline
    process_documents()
