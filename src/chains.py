from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from src.config import GROQ_API_KEY, LLM_MODEL
from src.prompts import (
    DOCUMENT_PROMPT,
    HISTORY_PROMPT,
    GENERAL_PROMPT,
)


# Initialize Groq LLM
llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2
)


# Document-based chain
document_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=DOCUMENT_PROMPT
)

document_chain = document_prompt | llm


# Conversation history chain
history_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template=HISTORY_PROMPT
)

history_chain = history_prompt | llm


# General conversation chain
general_prompt = PromptTemplate(
    input_variables=["question"],
    template=GENERAL_PROMPT
)

general_chain = general_prompt | llm


def get_document_response(context: str, question: str):
    response = document_chain.invoke({
        "context": context,
        "question": question
    })
    return response.content


def get_history_response(history: str, question: str):
    response = history_chain.invoke({
        "history": history,
        "question": question
    })
    return response.content


def get_general_response(question: str):
    response = general_chain.invoke({
        "question": question
    })
    return response.content