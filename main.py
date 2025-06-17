import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Config
index_name = "test"
region = "us-east-1"
cloud = "aws"
dimension = 384
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"🔌 Initializing Pinecone in cloud: {cloud}, region: {region}")
print(f"📊 Using index: {index_name}")

# Step 1: Initialize Pinecone v3
pc = Pinecone(api_key=pinecone_api_key)

# Step 2: Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print("📁 Index not found. Creating...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )
    print("✅ Index created.")

index = pc.Index(index_name)
print("✅ Connected to index.")

# Step 3: Load embedding model
print("🔄 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 4: Connect LangChain to Pinecone
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="text",
    pinecone_api_key=pinecone_api_key,
    pinecone_environment=region
)

# Step 5: Prompt Template
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

# Step 6: Setup QA Chain
def create_qa_chain():
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OpenAI(
        temperature=0.7,
        max_tokens=512,
        openai_api_key=openai_api_key
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

# Step 7: Optional - Add Sample Data
def add_sample_data():
    texts = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a branch of artificial intelligence that uses data to learn patterns.",
        "Pinecone is a vector database used for similarity search in AI systems.",
        "LangChain is a framework for building applications with language models.",
        "NLP enables computers to understand and generate human language.",
        "Vector embeddings are used to represent semantic meaning in high dimensions.",
        "OpenAI provides APIs for language models like GPT-4.",
        "Data science involves data analysis, machine learning, and data visualization."
    ]
    vectors = []
    for i, text in enumerate(texts):
        vectors.append({
            "id": f"doc_{i}",
            "values": embeddings.embed_query(text),
            "metadata": {"text": text, "source": f"sample_{i}"}
        })
    index.upsert(vectors=vectors)
    print(f"✅ Added {len(vectors)} sample documents.")

# Step 8: Main Loop
def main():
    try:
        stats = index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            print("📝 Index is empty. Uploading sample data...")
            add_sample_data()

        print("⚙️ Creating QA chain...")
        qa_chain = create_qa_chain()

        print("✅ Ready to answer questions.")
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
