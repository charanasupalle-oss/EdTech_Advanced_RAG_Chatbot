
import os

from dotenv import load_dotenv
from supabase import create_client

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from src.chains import get_document_response


load_dotenv()


# Initialize LLM
router_llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0
)


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


# Fixed refusal response
REFUSAL = (
    "I couldn't find this information in my uploaded "
    "learning documents."
)


# History prompt
HISTORY_PROMPT = PromptTemplate(
    input_variables=["history", "question"],
    template="""Answer the user's question ONLY using the conversation history below.

Do not use outside knowledge.
Do not invent or assume information.
If the requested information is not present in the conversation,
reply: "I couldn't find this information in our conversation."

Conversation history:
{history}

User question:
{question}

Answer:"""
)


# Detect greetings
def is_greeting(question: str):

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "bye"
    ]

    return question.lower().strip() in greetings


# Detect conversation history questions
def is_history_question(question: str):

    history_keywords = [
        "my last question",
        "my previous question",
        "what did i ask",
        "what was i asking",
        "your last answer",
        "your previous answer",
        "repeat your answer",
        "repeat my question",
        "what did you just say",
        "earlier question",
        "previous message",
        "last message",
        "what did we discuss",
        "what did i tell you",
        "what have i asked",
        "summarize our conversation",
        "summarise our conversation",
        "in this conversation",
        "in our conversation"
    ]

    question_lower = question.lower().strip()

    return any(
        keyword in question_lower
        for keyword in history_keywords
    )


# Retrieve ONLY Python_Basics.pdf documents
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

        # Restrict source to Python_Basics.pdf
        source = str(metadata.get("source", ""))

        if "Python_Basics.pdf" not in source:
            continue

        similarity = row.get("similarity")

        doc = Document(
            page_content=row["content"],
            metadata={
                **metadata,
                "similarity": similarity
            }
        )

        retrieved_docs.append(doc)

    return retrieved_docs


# Handle conversation history questions
def get_history_response(question: str, history: str):

    if not history or not history.strip():

        return "I couldn't find this information in our conversation."

    prompt = HISTORY_PROMPT.format(
        history=history,
        question=question
    )

    response = router_llm.invoke(prompt)

    return response.content


# Main pipeline
def process_question(question: str, history: str = ""):

    # Handle greetings
    if is_greeting(question):

        return {
            "category": "GENERAL",
            "answer": (
                "Hello! 👋 Ask me questions about "
                "your Python learning document."
            ),
            "sources": []
        }

    # Handle conversation history questions
    if is_history_question(question):

        answer = get_history_response(
            question=question,
            history=history
        )

        return {
            "category": "HISTORY",
            "answer": answer,
            "sources": []
        }

    # All other questions MUST use the PDF
    retrieved_docs = retrieve_documents(
        question,
        k=3
    )

    # Reject when no PDF chunks are retrieved
    if not retrieved_docs:

        return {
            "category": "DOCUMENT",
            "answer": REFUSAL,
            "sources": []
        }

    # Reject weak retrieval results
    scores = [
        doc.metadata.get("similarity")
        for doc in retrieved_docs
        if doc.metadata.get("similarity") is not None
    ]

    if scores and max(scores) < 0.40:

        return {
            "category": "DOCUMENT",
            "answer": REFUSAL,
            "sources": []
        }

    # Build context from PDF chunks only
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Generate answer using PDF context only
    answer = get_document_response(
        context=context,
        question=question
    )

    return {
        "category": "DOCUMENT",
        "answer": answer,
        "sources": retrieved_docs
    }
