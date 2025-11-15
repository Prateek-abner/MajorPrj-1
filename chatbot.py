# -*- coding: utf-8 -*-
"""
Chatbot Backend - Loads vectorstore and answers questions
Run this AFTER data_processor.py
"""

import os
import sys
from typing import Dict
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch


# Configuration
CHROMA_DB_DIR = "./chroma_db"
os.environ.setdefault("GROQ_API_KEY", "xxxxxxxxxxxxxxxxxx")


# Prompt Template for Legal Questions
PROMPT_TEMPLATE = """You are an expert legal assistant specializing in Indian law. Use the following legal documents to answer the question accurately and professionally.

Context from legal documents:
{context}

Question: {question}

Instructions:
- Provide accurate answers based ONLY on the context provided
- Cite specific sections, acts, or clauses when relevant
- If the answer is not in the context, say "I don't have information about this in the provided legal documents"
- Use clear, professional legal language
- Be concise but thorough

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def load_vectorstore(persist_directory: str = CHROMA_DB_DIR):
    """Load existing vector store from disk."""
    
    if not os.path.exists(persist_directory):
        print(f"❌ ERROR: Vector database not found at '{persist_directory}'")
        print("📝 Please run data_processor.py first to create the database")
        sys.exit(1)
    
    print(f"📂 Loading vector database from '{persist_directory}'...")
    
    # Configure embeddings (must match data_processor.py)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32}
    )
    
    try:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Verify vectorstore
        collection_count = vectordb._collection.count()
        print(f"✅ Loaded vectorstore with {collection_count} embedding(s)")
        return vectordb
        
    except Exception as e:
        print(f"❌ ERROR loading vectorstore: {e}")
        print("📝 Try running data_processor.py again to rebuild the database")
        sys.exit(1)


def init_llm():
    """Initialize Groq LLM for question answering."""
    print("🤖 Initializing Groq LLM...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ ERROR: GROQ_API_KEY not found in environment")
        sys.exit(1)
    
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=4096
        )
        print("✅ LLM initialized successfully")
        return llm
        
    except Exception as e:
        print(f"❌ ERROR initializing LLM: {e}")
        sys.exit(1)


def build_qa_chain(vectordb, llm):
    """Build the RAG QA chain."""
    print("🔗 Building RAG chain...")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    print("✅ RAG chain ready")
    return qa_chain


def initialize_chatbot(persist_dir: str = CHROMA_DB_DIR):
    """Initialize the complete chatbot pipeline."""
    
    print("="*60)
    print("🏛️ LAW CHATBOT - Initialization")
    print("="*60)
    
    # Step 1: Load vectorstore
    vectordb = load_vectorstore(persist_dir)
    
    # Step 2: Initialize LLM
    llm = init_llm()
    
    # Step 3: Build QA chain
    qa_chain = build_qa_chain(vectordb, llm)
    
    print("\n" + "="*60)
    print("✅ CHATBOT READY!")
    print("="*60)
    
    return {"qa_chain": qa_chain, "vectordb": vectordb}


def ask_question(qa_chain, query: str) -> Dict:
    """Ask a question and get an answer with sources."""
    try:
        result = qa_chain({"query": query})
        return {
            "success": True,
            "answer": result.get("result", "No answer generated"),
            "source_count": len(result.get("source_documents", [])),
            "sources": result.get("source_documents", [])
        }
    except Exception as e:
        return {
            "success": False,
            "answer": f"Error processing question: {str(e)}",
            "source_count": 0,
            "sources": []
        }


if __name__ == "__main__":
    # Interactive CLI mode
    pipeline = initialize_chatbot()
    qa = pipeline["qa_chain"]
    
    print("\n🏛️ LAW CHATBOT - Interactive Mode")
    print("="*60)
    print("Type your legal questions below")
    print("Type 'exit' or 'quit' to stop")
    print("="*60)
    
    while True:
        q = input("\n💬 Your Question: ").strip()
        
        if q.lower() in ("exit", "quit", ""):
            print("\n👋 Goodbye!")
            break
        
        print("\n🔍 Searching legal documents...")
        result = ask_question(qa, q)
        
        if result["success"]:
            print(f"\n📖 Answer:\n{result['answer']}")
            print(f"\n📚 Found {result['source_count']} relevant source(s)")
        else:
            print(f"\n❌ {result['answer']}")

