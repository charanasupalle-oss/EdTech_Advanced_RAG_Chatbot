import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


load_dotenv()


# Load ONLY Python_Basics.pdf
def load_documents():
    pdf_file = Path(DATA_DIR) / "Python_Basics.pdf"

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"Python_Basics.pdf not found at: {pdf_file}"
        )

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    # Keep source metadata restricted to this PDF
    for document in documents:
        document.metadata["source"] = "Python_Basics.pdf"

    print(f"Loaded {len(documents)} pages from Python_Basics.pdf")

    return documents


# Split PDF pages into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")

    return chunks


# Store chunks and embeddings in Supabase
def create_vectorstore(chunks):

    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )

    # Clear previous records before inserting the single PDF
    supabase.table("documents").delete().neq("id", "0").execute()

    print("Cleared old documents from Supabase.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    texts = [chunk.page_content for chunk in chunks]

    vectors = embeddings.embed_documents(texts)

    records = []

    for chunk, vector in zip(chunks, vectors):
        records.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": vector
        })

    batch_size = 100

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        supabase.table("documents").insert(batch).execute()

        print(f"Uploaded {len(batch)} chunks to Supabase.")

    print("Python_Basics.pdf stored in Supabase successfully!")


# Main ingestion pipeline
def main():
    print("Starting ingestion of Python_Basics.pdf...")

    documents = load_documents()

    chunks = split_documents(documents)

    create_vectorstore(chunks)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()