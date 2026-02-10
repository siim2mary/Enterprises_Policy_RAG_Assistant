import os
import time
from dotenv import load_dotenv

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Modern LangChain Imports (2026 Standard) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Local & Free
from langchain_chroma import Chroma # Optimized Vector Store
from langchain_groq import ChatGroq # High-speed LLM

# Chains moved to langchain_classic in latest versions
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load .env (Ensure GROQ_API_KEY is present)
load_dotenv()

# --- Configuration ---
DATA_PATH = "data/"
DB_PATH = "vectorstore_policy"
# Using a SOTA local embedding model (No API costs/limits)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Recommended Llama 3.1 model for speed and accuracy
LLM_MODEL = "llama-3.1-8b-instant"

def get_embeddings():
    """Returns local HuggingFace embeddings running on your CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'}
    )

def ingest_docs():
    """Processes PDFs in /data and builds the local vector database."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Folder '{DATA_PATH}' created. Please add your PDFs.")
        return

    print("📖 Loading policy documents from disk...")
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())

    if not docs:
        print("⚠️ No PDFs found. Please add files to the /data folder.")
        return

    # Split documents into chunks to fit LLM context windows
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    print(f"📡 Vectorizing {len(chunks)} chunks locally...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_PATH
    )
    print("✅ Local Vector Database built successfully.")

def ask_question(query):
    """Retrieves context and generates an answer using Groq's Llama 3.1."""
    # 1. Load the database
    vector_db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=get_embeddings()
    )
    
    # 2. Initialize Groq LLM
    llm = ChatGroq(
        model_name=LLM_MODEL,
        temperature=0.1, # Low temperature for factual policy adherence
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 3. Create the System Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a professional Enterprise Policy Assistant. 
    Use the provided context to answer the user's question accurately.
    If the answer is not in the context, say: "I don't know based on the documents provided."
    
    Context: {context}
    Question: {input}
    
    Final Answer:""")

    # 4. Assemble the RAG Chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain)

    print(f"⚡ Searching policies via Groq...")
    try:
        start_time = time.time()
        response = retrieval_chain.invoke({"input": query})
        end_time = time.time()
        
        print(f"\n🤖 Assistant: {response['answer']}")
        print(f"⏱️ Response time: {round(end_time - start_time, 2)} seconds")
    except Exception as e:
        print(f"❌ Error connecting to Groq: {e}")

if __name__ == "__main__":
    # Ensure database exists
    if not os.path.exists(DB_PATH):
        ingest_docs()
    
    print("\n--- 📚 Enterprise Policy Assistant Online ---")
    print("(Type 'exit' to quit)\n")
    
    while True:
        user_input = input("Ask a policy question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.strip():
            ask_question(user_input)