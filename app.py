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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"

# Configuration
index_name = "test"
dimension = 384
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
data_dir = "data/"  # Directory where your PDFs are stored

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# PDF Loader + Splitter
def load_and_split_pdfs(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' not found.")

    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDFs found in '{directory_path}'.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(documents)

# Embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

# Upload documents to Pinecone
def store_in_pinecone(chunks, embeddings):
    existing_indexes = pc.list_indexes().names()

    if index_name in existing_indexes:
        index_info = pc.describe_index(index_name)
        if index_info.dimension != dimension:
            pc.delete_index(index_name)
            time.sleep(5)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=pinecone_env)
            )
    else:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_env)
        )

    index = pc.Index(index_name)
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    embedded_texts = embeddings.embed_documents(texts)

    vectors = []
    for i, (text, embedding_vec, metadata) in enumerate(zip(texts, embedded_texts, metadatas)):
        vectors.append({
            "id": f"doc_{i}",
            "values": embedding_vec,
            "metadata": {**metadata, "text": text}
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    return PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Prompt template
prompt_template = """
You are an expert summarizer. Use the following context to write a detailed and concise summary of the concept mentioned in the question. 
If the context is not sufficient, say "I need more context to summarize this topic."

Context:
{context}

Question: {question}

Summary:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# QA Chain setup
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

# Streamlit App
st.set_page_config(page_title=" PDF Q&A with Pinecone + Cohere", layout="wide")
st.title(" Ask Questions from PDFs using Pinecone + Cohere")
st.markdown("Upload your PDFs to the `data/` directory and ask questions based on the content.")

if "qa_chain" not in st.session_state:
    with st.spinner(" Loading PDFs and creating vector index..."):
        try:
            chunks = load_and_split_pdfs(data_dir)
            embeddings = get_embeddings()
            vectorstore = store_in_pinecone(chunks, embeddings)
            qa_chain = create_qa_chain(vectorstore)
            st.session_state.qa_chain = qa_chain
            st.success(" Data loaded and index created!")
        except Exception as e:
            st.error(f" Error: {e}")
            st.stop()

# Input UI
question = st.text_input("Ask a question from the uploaded PDFs:")

if question:
    with st.spinner(" Searching and generating answer..."):
        try:
            response = st.session_state.qa_chain.invoke({"query": question})
            st.subheader("Answer:")
            st.write(response["result"])

            if response.get("source_documents"):
                st.subheader("Source Documents:")
                for i, doc in enumerate(response["source_documents"], 1):
                    with st.expander(f"Document {i}"):
                        st.write(doc.page_content)
        except Exception as e:
            st.error(f" Error: {e}")
