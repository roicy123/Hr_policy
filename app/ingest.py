from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    # normalize -> unit vectors, so distance maps cleanly to cosine similarity
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

def load_docs(folder="data/policies"):
    docs = []
    for path in Path(folder).glob("*"):
        if path.suffix == ".pdf":
            docs += PyPDFLoader(str(path)).load()
        elif path.suffix == ".txt":
            docs += TextLoader(str(path), encoding="utf-8").load()
    return docs

if __name__ == "__main__":
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    FAISS.from_documents(chunks, get_embeddings()).save_local("vectorstore")