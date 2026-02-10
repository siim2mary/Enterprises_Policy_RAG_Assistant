import os
import streamlit as st
from dotenv import load_dotenv

# --- Modern LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load Environment Variables
load_dotenv()

# --- Configuration ---
DATA_PATH = "data/"
DB_PATH = "vectorstore_policy"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.1-8b-instant"

# Page Config
st.set_page_config(page_title="Enterprise Policy Assistant", page_icon="📚")
st.title("📚 Enterprise Policy Assistant")
st.markdown("Ask questions about your company's HR and Security policies.")

# --- Utility Functions ---
@st.cache_resource
def load_vector_db():
    """Initializes embeddings and loads the vector database."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    # Check if DB exists, if not, trigger ingestion
    if not os.path.exists(DB_PATH):
        with st.spinner("Initializing database for the first time..."):
            ingest_docs(embeddings)
    
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def ingest_docs(embeddings):
    """Processes PDFs and creates the vector database."""
    if not os.path.exists(DATA_PATH) or not any(f.endswith(".pdf") for f in os.listdir(DATA_PATH)):
        st.error(f"Please add PDF files to the `{DATA_PATH}` folder first!")
        st.stop()

    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

# --- Chat Interface ---
# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is the policy on..."):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process RAG Response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                # 1. Setup Chain
                vector_db = load_vector_db()
                llm = ChatGroq(
                    model_name=LLM_MODEL,
                    temperature=0.1,
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
                
                system_prompt = ChatPromptTemplate.from_template("""
                You are a professional Enterprise Policy Assistant. 
                Use the provided context to answer the user's question accurately.
                If the answer is not in the context, say: "I don't know based on the documents provided."
                
                Context: {context}
                Question: {input}
                
                Final Answer:""")

                combine_docs_chain = create_stuff_documents_chain(llm, system_prompt)
                retrieval_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain)

                # 2. Invoke Chain
                response = retrieval_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                # 3. Show Answer
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")