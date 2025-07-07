# app1.py

import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere
from video_to_pdf import download_video_and_generate_pdf, convert_video_to_pdf

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"
cohere_api_key = os.getenv("COHERE_API_KEY")

# Configuration
index_name = "video-qa"
dimension = 384
data_dir = "data/"
os.makedirs(data_dir, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Functions
def load_and_split_pdfs(directory_path):
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDFs found in '{directory_path}'.")
    logger.info(f"Loaded {len(documents)} documents.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(documents)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store_in_pinecone(chunks, embeddings):
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        index_info = pc.describe_index(index_name)
        if index_info.dimension != dimension:
            pc.delete_index(index_name)
            time.sleep(5)
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_env)
        )

    index = pc.Index(name=index_name)
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    embedded_texts = embeddings.embed_documents(texts)

    vectors = [{
        "id": f"doc_{i}",
        "values": vec,
        "metadata": {**meta, "text": text}
    } for i, (text, vec, meta) in enumerate(zip(texts, embedded_texts, metadatas))]

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
        logger.info(f"Uploaded batch {i//batch_size + 1} to Pinecone")

    return PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Prompt template
prompt_template = """
You are an expert summarizer. Use the following context to answer the user's question. 
If the context is not sufficient, say "I need more context."

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatCohere(
        model="command-r",
        temperature=0.7,
        max_tokens=512,
        api_key=cohere_api_key
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

# Streamlit UI
st.set_page_config(page_title="🎥 Video Q&A App", layout="wide")
st.title("🎥 Video Transcript Q&A")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "webm"])
video_url = st.text_input("Or paste a YouTube or direct video link:")

if video_file or video_url:
    with st.spinner("Transcribing and processing..."):
        try:
            if video_url:
                pdf_path = download_video_and_generate_pdf(video_url)
            else:
                video_path = os.path.join(data_dir, video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.read())
                pdf_path = convert_video_to_pdf(video_path, video_path.replace(".mp4", ".pdf"))

            st.success(f"Transcript saved at: {pdf_path}")
            chunks = load_and_split_pdfs(data_dir)
            embeddings = get_embeddings()
            vectorstore = store_in_pinecone(chunks, embeddings)
            qa_chain = create_qa_chain(vectorstore)
            st.session_state.qa_chain = qa_chain
            st.success("Q&A system is ready!")
        except Exception as e:
            st.error(f"Error during processing: {e}")

# Ask questions
question = st.text_input("Ask a question from the video:")
if question and "qa_chain" in st.session_state:
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.qa_chain.invoke({"query": question})
            st.subheader("Answer:")
            st.write(result["result"])

            if result.get("source_documents"):
                st.subheader("Context from Transcript:")
                for i, doc in enumerate(result["source_documents"], 1):
                    with st.expander(f"Source {i}"):
                        st.write(doc.page_content)
        except Exception as e:
            st.error(f"Error: {e}")
