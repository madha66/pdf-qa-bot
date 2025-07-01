import os
import tempfile
import streamlit as st
import fitz 
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
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap
    ).split_text(text)

def embed_text_with_chroma(chunks):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(
        chunks, embedding=embedder
    )
def get_llm_chain():
    os.environ["GROQ_API_KEY"]="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"
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
st.set_page_config(page_title="PDF Q/A", layout="centered")
st.title("📚 Chat with your PDF")

if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.vectorstore = None
    st.session_state.chain = None

file = st.file_uploader("Upload a PDF", type="pdf")

if file and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF…"):
        text = extract_text_from_pdf(file)
        chunks = split_text(text)
        st.session_state.vectorstore = embed_text_with_chroma(chunks)
        st.session_state.chain = get_llm_chain()
    st.success("Ready! Ask a question ↓")

with st.form("question-form"):
    q = st.text_input("Your question:")
    submit=st.form_submit_button("Ask")
    if submit and q:
        with st.spinner("Thinking…"):
            a = get_answer(st.session_state.vectorstore, q, st.session_state.chain)
            if not a or a.lower() in ["i don't know","i cannot answer that",""]:
                a="Sorry, I couldn’t find the answer in the document"
        st.session_state.chat.append((q, a))

for q, a in st.session_state.chat:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
if st.session_state.chat:
    st.markdown("download chat")
    docx=createdocx(st.session_state.chat)
    st.markdown(filedownload(docx,"chat.docx","Download as .docx"),unsafe_allow_html=True)


