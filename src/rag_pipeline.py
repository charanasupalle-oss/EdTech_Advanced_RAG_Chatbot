import os

from dotenv import load_dotenv
from supabase import create_client

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from src.chains import (
    get_document_response,
    get_history_response,
    get_general_response,
)


load_dotenv()


# Initialize LLM for question routing
router_llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0
)


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


# Retrieve documents from Supabase
def retrieve_documents(question: str, k: int = 3):

    query_embedding = embeddings.embed_query(question)

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": k
        }
    ).execute()

    retrieved_docs = []

    for row in response.data:

        metadata = row.get("metadata") or {}

        metadata["similarity"] = row.get("similarity")

        doc = Document(
            page_content=row["content"],
            metadata=metadata
        )

        retrieved_docs.append(doc)

    return retrieved_docs


# Classify user question
def route_question(question: str, history: str = ""):

    routing_prompt = f"""
    Classify the user's question into exactly one category:

    DOCUMENT: Questions about educational documents, learning,
    technical concepts, study material, skills or career topics
    that may require knowledge retrieval.

    HISTORY: Follow-up questions that depend on previous conversation.

    GENERAL: Greetings, casual conversation or questions
    that do not require document retrieval.

    Conversation History:
    {history}

    User Question:
    {question}

    Return only one category:
    DOCUMENT, HISTORY or GENERAL.
    """

    response = router_llm.invoke(routing_prompt)
    category = response.content.strip().upper()

    if "HISTORY" in category and history.strip():
        return "HISTORY"

    if "DOCUMENT" in category:
        return "DOCUMENT"

    return "GENERAL"


# Main RAG pipeline
def process_question(question: str, history: str = ""):

    category = route_question(question, history)

    # DOCUMENT route
    if category == "DOCUMENT":

        retrieved_docs = retrieve_documents(
            question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        answer = get_document_response(
            context=context,
            question=question
        )

        return {
            "category": "DOCUMENT",
            "answer": answer,
            "sources": retrieved_docs
        }

    # HISTORY route
    elif category == "HISTORY":

        answer = get_history_response(
            history=history,
            question=question
        )

        return {
            "category": "HISTORY",
            "answer": answer,
            "sources": []
        }

    # GENERAL route
    else:

        answer = get_general_response(question)

        return {
            "category": "GENERAL",
            "answer": answer,
            "sources": []
        }