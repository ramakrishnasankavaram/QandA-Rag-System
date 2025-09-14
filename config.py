import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Google AI API Key
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    
    # Gemini Model Configuration
    GEMINI_MODEL = "gemini-1.5-flash"
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Directories
    DATA_DIR = "data/documents"
    VECTOR_STORE_DIR = "vectorstore"
    
    # Streamlit Configuration
    PAGE_TITLE = "RAG Q&A System"

    PAGE_ICON = "🤖"
