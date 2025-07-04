import os
import tempfile
import streamlit as st
import fitz 
import hashlib
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from docx import Document
from docx2pdf import convert
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def split_text(text, size=1000, overlap=100):
    splitted=RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap
    )
    return splitted.split_text(text)

def embed_text_with_faiss(chunks,index_path):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vstore=FAISS.from_texts(
        chunks, embedding=embedder
    )
    vstore.save_local(index_path)
    return vstore
def get_llm_chain():
    groq_api_key=os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Api Key not Found")
    os.environ["GROQ_API_KEY"]=groq_api_key
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=300, 
    )
    return llm
def get_answer(vstore, query, chain):
    docs = vstore.as_retriever(search_kwargs={"k":3})
    qa_bot=RetrievalQA.from_chain_type(
        llm=chain,
        retriever=docs,
        chain_type="stuff",
        return_source_documents=False
    )
    result=qa_bot({"query":query})
    answer=result['result'].strip()
    return answer
def createdocx(chat):
    doc=Document()
    doc.add_heading("Chatting with bot",0)
    for q,a in chat:
        doc.add_paragraph(f"you: {q}",style='List Bullet')
        doc.add_paragraph(f"bot: {a}",style="List Bullet")
        doc.add_paragraph(" ")
    docx=tempfile.NamedTemporaryFile(delete=False,suffix=".docx")
    doc.save(docx.name)
    return docx.name
def filedownload(filepath,filename,label):
    with open(filepath,"rb") as f:
        bytes_data=f.read()
        Bytes=base64.b64encode(bytes_data).decode()
        href= f'<a href="data:application/octet-stream;base64,{Bytes}" download = "{filename}">{label}</a>'
        return href
def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()
st.set_page_config(page_title="PDF Q/A", layout="centered")
st.title("📚 Chat with your PDF")
if "chat" not in st.session_state:
    st.session_state.chat = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = ""
file = st.file_uploader("Upload a PDF", type="pdf")
if file:
    file_hash=get_file_hash(file)
    index_dir=os.path.join(tempfile.gettempdir(),f"faiss_{file_hash}")
    if st.session_state.last_file_hash!=file_hash:
        with st.spinner("Processing PDF…"):
            st.session_state.chat = []
            if os.path.exists(index_dir):
                vstore=FAISS.load_local(index_dir, embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
            else:
                text = extract_text_from_pdf(file)
                if not text.split():
                    st.warning("The uploaded PDF contains no readable text.")
                    return 
                else:
                    chunks = split_text(text)
                    if not chunks:
                        st.warning("The PDF is read,but there can be no chunks created")
                        return
                    else:
                        vstore=embed_text_with_faiss(chunks,index_dir)
        st.session_state.vectorstore = vstore
        st.session_state.chain = get_llm_chain()
        st.session_state.last_file_hash=file_hash
    st.success("Ready! Ask a question ↓")
if st.session_state.vectorstore and st.session_state.chain:
    with st.form("question-form"):
        if not file:
            q = st.text_input("Your question:",disabled=True)
            submit = st.form_submit_button("Ask",disabled=True)
        else:
            q = st.text_input("Your question:",disabled=False)
            submit = st.form_submit_button("Ask",disabled=False)
        if submit and q:
            with st.spinner("Thinking…"):
                a = get_answer(st.session_state.vectorstore, q, st.session_state.chain)
                if not a or a.lower() in ["i don't know", "i cannot answer that", ""]:
                    a = "Sorry, I couldn’t find the answer in the document."
                st.session_state.chat.append((q, a))
                with st.container():
                    st.markdown(f"**You:** {q}")
                    st.markdown(f"**Bot:** {a}")
                    st.markdown("---")
    if st.session_state.chat:
        st.markdown("Download Chat")
        docx = createdocx(st.session_state.chat)
        st.markdown(filedownload(docx,"chat.docx","Download as .docx"),unsafe_allow_html=True)
else:
    st.warning("Please upload a document to chat with the bot.")


