import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Configuration
index_name = "test"
dimension = 384
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"

# API Keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Pinecone
print(f"🔌 Connecting to Pinecone in environment: {pinecone_env}")
pc = Pinecone(api_key=pinecone_api_key)

# Connect to existing index
if index_name not in pc.list_indexes().names():
    raise ValueError(f"❌ Index '{index_name}' not found in Pinecone. Please create it first.")
index = pc.Index(index_name)
print("✅ Connected to existing Pinecone index.")

# Load embeddings
print("🔄 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Setup vectorstore connection
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"  # This must match the metadata field used when uploading data
)

# Prompt template for QA
prompt_template = """
Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# QA Chain setup
def create_qa_chain():
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

# Main CLI Loop
def main():
    try:
        print("⚙️ Setting up QA chain...")
        qa_chain = create_qa_chain()

        print("✅ Ready to answer your questions.")
        while True:
            query = input("\n❓ Your question (type 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not query:
                print("⚠️ Please enter a valid question.")
                continue

            print("🤔 Thinking...")
            response = qa_chain.invoke({"query": query})

            print("\n📝 Answer:")
            print(response["result"])

            if response.get("source_documents"):
                print("\n📚 Source Documents:")
                for i, doc in enumerate(response["source_documents"], 1):
                    preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    print(f"\n--- Document {i} ---")
                    print(preview)
            else:
                print("\n📚 No source documents found.")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
