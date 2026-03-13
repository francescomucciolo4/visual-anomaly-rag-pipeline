from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

KB_PATH = "knowledge_base/knowledge_base.md"
CHROMA_DIR = "chroma_db"

def build_index():
    # Carica il file .md
    loader = TextLoader(KB_PATH, encoding="utf-8")
    documents = loader.load()
    
    # Splitta in chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Crea embeddings e salva in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print(f"✓ Index saved in '{CHROMA_DIR}/'")
    return vectorstore

if __name__ == "__main__":
    build_index()