import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore

# Step 1: Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = "test"
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
# OR
# embedding_model_name = "BAAI/bge-large-en-v1.5"  # 1024 dimensions
dimension = 768  # Update this to match your chosen model
region = "us-east-1"

if not api_key:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env")

# Step 2: Initialize Pinecone v3+ client
pc = Pinecone(api_key=api_key)

# Step 3: Create index if it doesn't exist (or recreate with correct dimensions)
existing_indexes = pc.list_indexes().names()
if index_name in existing_indexes:
    # Check if dimensions match
    index_info = pc.describe_index(index_name)
    existing_dimension = index_info.dimension
    
    if existing_dimension != dimension:
        print(f"⚠️ Index '{index_name}' exists with dimension {existing_dimension}, but we need {dimension}")
        print(f"🗑️ Deleting existing index...")
        pc.delete_index(index_name)
        
        # Wait a moment for deletion to complete
        import time
        time.sleep(5)
        
        print(f"🛠 Creating new index '{index_name}' with dimension {dimension}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region)
        )
    else:
        print(f"✅ Index '{index_name}' already exists with correct dimension {dimension}.")
else:
    print(f"🛠 Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region)
    )

# Step 4: PDF Loader + Splitter
def load_and_split_pdfs(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' not found.")
    
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDFs found in '{directory_path}'.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(documents)

# Step 5: Embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 6: Upload to Pinecone
def store_in_pinecone(chunks, embeddings):
    # Get the index object
    index = pc.Index(index_name)
    
    print("🔄 Using manual embedding and upload approach...")
    
    # Extract texts and metadata from chunks
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    
    # Generate embeddings for all texts
    print("🧠 Generating embeddings for all chunks...")
    embedded_texts = embeddings.embed_documents(texts)
    
    # Prepare vectors for upsert
    vectors = []
    for i, (text, embedding_vec, metadata) in enumerate(zip(texts, embedded_texts, metadatas)):
        vectors.append({
            "id": f"doc_{i}",
            "values": embedding_vec,
            "metadata": {**metadata, "text": text}
        })
    
    # Upload in batches
    batch_size = 100
    total_batches = (len(vectors) - 1) // batch_size + 1
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"📤 Uploaded batch {i//batch_size + 1}/{total_batches}")
    
    # Create a simple vectorstore class for querying
    class SimplePineconeStore:
        def __init__(self, index, embeddings):
            self.index = index
            self.embeddings = embeddings
        
        def similarity_search(self, query, k=4):
            # Embed the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            # Convert results to documents
            documents = []
            for match in results['matches']:
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')
                documents.append(type('Document', (), {'page_content': text, 'metadata': metadata})())
            
            return documents
    
    # Return the custom vectorstore
    vectorstore = SimplePineconeStore(index, embeddings)
    return vectorstore

# Step 7: Main Execution
if __name__ == "__main__":
    try:
        print("📂 Loading and splitting PDFs...")
        chunks = load_and_split_pdfs("data/")
        print(f"✅ Loaded and split into {len(chunks)} chunks.")

        print("🧠 Generating embeddings...")
        embeddings = get_embeddings()

        print("📤 Uploading to Pinecone...")
        vectorstore = store_in_pinecone(chunks, embeddings)

        print("🎉 Done! Data successfully stored in Pinecone.")
        
        # Optional: Test a simple query
        print("\n🔍 Testing similarity search...")
        test_query = "What is the main topic?"
        results = vectorstore.similarity_search(test_query, k=3)
        print(f"Found {len(results)} similar documents for test query.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()