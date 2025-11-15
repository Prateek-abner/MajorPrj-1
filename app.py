# -*- coding: utf-8 -*-
"""
Streamlit Frontend for Law Chatbot
Run this AFTER data_processor.py has been executed

IMPORTANT: Run with 'streamlit run app.py' NOT 'python app.py'
"""

import streamlit as st
import os
import sys

# Check if running with streamlit command
def check_streamlit_runtime():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return False
        return True
    except:
        return False

if not check_streamlit_runtime():
    print("\n" + "="*60)
    print("❌ ERROR: Incorrect Run Command")
    print("="*60)
    print("\nYou ran: python app.py")
    print("This is WRONG for Streamlit apps!\n")
    print("✅ Correct command:")
    print("   streamlit run app.py")
    print("\nOr alternatively:")
    print("   python -m streamlit run app.py")
    print("="*60)
    sys.exit(1)

from chatbot import initialize_chatbot, ask_question

# ----- Improved CSS -----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #DBEAFE 0%, #BFDBFE 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #F0F9FF;
        color: #1E293B;             /* Strong dark color for answer text */
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
        font-size: 1.15rem;         /* Larger text for clarity */
        font-weight: 500;
        letter-spacing: 0.03em;
    }
    .source-box {
        background-color: #FFFBEB;
        color: #1E293B;             /* Makes sources readable */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .example-question {
        background-color: #F3F4F6;
        color: #1E293B;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .example-question:hover {
        background-color: #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)
# ----- End CSS -----

@st.cache_resource
def load_chatbot():
    try:
        return initialize_chatbot()
    except SystemExit:
        return None
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Law Chatbot - Indian Legal Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("📚 About")
        st.info(
            "This chatbot provides answers to legal questions based on "
            "Indian law documents uploaded to the system. It uses RAG "
            "(Retrieval-Augmented Generation) to provide accurate responses."
        )
        
        st.title("🔧 System Status")
        if os.path.exists("./chroma_db"):
            db_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk("./chroma_db")
                for filename in filenames
            ) / (1024 * 1024)
            st.success(f"✅ Vector Database: Ready ({db_size:.1f} MB)")
        else:
            st.error("❌ Vector Database: Not Found")
            st.warning("Please run `python data_processor.py` first!")
        
        st.title("💡 Example Questions")
        example_questions = [
            "What are the fundamental rights in India?",
            "Explain Section 498A IPC",
            "What is the process for filing a civil suit?",
            "Explain the doctrine of basic structure",
            "What are the grounds for divorce in India?",
            "What is Article 21 of the Constitution?",
            "Explain the Hindu Succession Act",
            "What are the provisions of Special Marriage Act?"
        ]
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}", use_container_width=True):
                st.session_state.example_question = question
        
        st.title("⚙️ Settings")
        show_sources = st.checkbox("Show source documents", value=True)
        num_sources = st.slider("Number of sources to retrieve", 3, 10, 5)
        
        st.title("📊 Statistics")
        if "messages" in st.session_state:
            st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    if not os.path.exists("./chroma_db"):
        st.error("❌ Vector database not found!")
        st.warning("**Please follow these steps:**")
        st.code("""
1. Add PDF files to the 'data' folder
2. Run: python data_processor.py
3. Then run: streamlit run app.py
        """)
        st.info("**Need help?** Make sure you have PDF documents in the `data` folder before running the data processor.")
        st.stop()
    
    with st.spinner("🔄 Loading chatbot... Please wait..."):
        pipeline = load_chatbot()
    
    if pipeline is None:
        st.error("❌ Failed to initialize chatbot. Please check the logs and try again.")
        st.stop()
    
    qa_chain = pipeline["qa_chain"]
    st.success("✅ Chatbot loaded successfully!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "example_question" in st.session_state:
        example_q = st.session_state.example_question
        del st.session_state.example_question
        st.session_state.messages.append({"role": "user", "content": example_q})
        with st.spinner("🔍 Searching legal documents..."):
            result = ask_question(qa_chain, example_q)
        if result["success"]:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["source_count"]
            })
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and show_sources and "sources" in message:
                if message["sources"] > 0:
                    with st.expander(f"📚 View {message['sources']} source(s)"):
                        st.caption("Information retrieved from legal documents")
    
    if prompt := st.chat_input("Ask your legal question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching legal documents..."):
                result = ask_question(qa_chain, prompt)
            if result["success"]:
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                if show_sources and result["source_count"] > 0:
                    with st.expander(f"📚 View {result['source_count']} source(s)"):
                        st.caption("Information retrieved from legal documents")
                        for idx, doc in enumerate(result["sources"][:num_sources], 1):
                            source_text = doc.page_content[:400]
                            source_file = doc.metadata.get('source', 'Unknown')
                            page_num = doc.metadata.get('page', 'N/A')
                            st.markdown(
                                f'<div class="source-box">'
                                f'<b>📄 Source {idx}</b><br>'
                                f'<small><b>File:</b> {os.path.basename(source_file)} | <b>Page:</b> {page_num}</small><br><br>'
                                f'{source_text}...'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["source_count"]
                })
            else:
                st.error(result["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {result['answer']}",
                    "sources": 0
                })
    
    if st.session_state.messages:
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()
