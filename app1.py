import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
import hashlib
import base64

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from docx import Document


# ---------------------------- PDF to Text ----------------------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)


# ---------------------------- Text Splitter ----------------------------
def split_text(text, size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


# ---------------------------- Embeddings and FAISS ----------------------------
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"device": "cpu"}
    )


def embed_text_with_faiss(chunks, index_path, embedder):
    vstore = FAISS.from_texts(chunks, embedding=embedder)
    FAISS.save_local(index_path, vstore)
    return vstore


# ---------------------------- LLM Initialization ----------------------------
def get_llm_chain():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    return ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=300
    )


# ---------------------------- QA Chain ----------------------------
def get_answer(vstore, query, llm):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )
    result = qa({"query": query})
    return result["result"].strip()


# ---------------------------- Chat Export ----------------------------
def createdocx(chat):
    doc = Document()
    doc.add_heading("Chat with PDF Bot", 0)
    for q, a in chat:
        doc.add_paragraph(f"You: {q}", style='List Bullet')
        doc.add_paragraph(f"Bot: {a}", style='List Bullet')
        doc.add_paragraph(" ")
    docx_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(docx_file.name)
    return docx_file.name


def filedownload(filepath, filename, label):
    with open(filepath, "rb") as f:
        bytes_data = f.read()
    encoded = base64.b64encode(bytes_data).decode()
    return f'<a href="data:application/octet-stream;base64,{encoded}" download="{filename}">{label}</a>'


# ---------------------------- File Hashing ----------------------------
def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="PDF Q/A Bot", layout="centered")
st.title("📚 Chat with your PDF")

# Session State Initialization
if "chat" not in st.session_state:
    st.session_state.chat = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = ""

# File Upload
file = st.file_uploader("Upload a PDF file", type="pdf")

if file:
    file_hash = get_file_hash(file)
    index_dir = os.path.join(tempfile.gettempdir(), f"faiss_{file_hash}")

    if st.session_state.last_file_hash != file_hash:
        with st.spinner("Processing PDF..."):
            st.session_state.chat = []
            embedder = get_embedder()

            if os.path.exists(index_dir):
                vstore = FAISS.load_local(
                    index_dir, embeddings=embedder,
                    allow_dangerous_deserialization=True
                )
            else:
                text = extract_text_from_pdf(file)
                if not text.strip():
                    st.warning("The uploaded PDF contains no readable text.")
                    st.stop()
                chunks = split_text(text)
                if not chunks:
                    st.warning("Unable to split the PDF into chunks.")
                    st.stop()
                vstore = embed_text_with_faiss(chunks, index_dir, embedder)

            st.session_state.vectorstore = vstore
            st.session_state.chain = get_llm_chain()
            st.session_state.last_file_hash = file_hash
        st.success("PDF processed. You can now ask questions!")

# Q/A Section
if st.session_state.vectorstore and st.session_state.chain:
    with st.form("question-form"):
        q = st.text_input("Your question:", disabled=not file)
        submit = st.form_submit_button("Ask", disabled=not file)
        if submit and q:
            with st.spinner("Thinking..."):
                a = get_answer(st.session_state.vectorstore, q, st.session_state.chain)
                if not a or a.lower() in ["i don't know", "i cannot answer that", ""]:
                    a = "Sorry, I couldn’t find the answer in the document."
                st.session_state.chat.append((q, a))
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
                st.markdown("---")

# Chat Download
if st.session_state.chat:
    st.markdown("#### Download Chat Transcript")
    docx_path = createdocx(st.session_state.chat)
    st.markdown(filedownload(docx_path, "chat.docx", "📄 Download as .docx"), unsafe_allow_html=True)
else:
    st.info("Upload a PDF to start chatting with it.")
