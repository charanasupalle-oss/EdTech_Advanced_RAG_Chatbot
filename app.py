


import streamlit as st
from src.rag_pipeline import process_question

# Page configuration
st.set_page_config(
    page_title="EdTech AI Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #10243a;
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Top Header */
.top-header {
    background-color: #000000;
    padding: 15px 25px;
    display: flex;
    align-items: center;
    gap: 14px;
    border-radius: 10px;
    margin-bottom: 25px;
}

.logo-icon {
    font-size: 32px;
}

.brand-name {
    color: white !important;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
}

/* Main heading */
.chat-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #17365d;
}

.chat-subtitle {
    color: #64748b;
    font-size: 1rem;
}

/* Chat message container */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 12px;
}

/* Force chatbot reply text to black */
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
    color: black !important;
}

/* Chatbot reply links */
div[data-testid="stChatMessage"] a {
    color: #1565c0 !important;
}

/* Chatbot code text */
div[data-testid="stChatMessage"] code {
    color: black !important;
}

/* Chatbot source expander text */
[data-testid="stExpander"] * {
    color: black;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("🎓 EdTech AI")
    st.markdown("---")
    st.markdown("### Learning Assistant")
    st.caption("Ask questions about your learning documents.")
    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Powered by**")
    st.markdown("LangChain · Groq · Supabase")


# Top Black Header
st.markdown("""
<div class="top-header">
    <div class="logo-icon">🎓</div>
    <div class="brand-name">EDUTECH</div>
</div>
""", unsafe_allow_html=True)


# Main heading
st.markdown(
    '<div class="chat-title">EdTech Learning Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="chat-subtitle">Ask questions, learn concepts, and explore your uploaded study materials.</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 **Welcome!** I'm your EdTech Learning Assistant.\n\n"
            "Ask questions about Python using your uploaded "
            "learning document, or ask about our conversation."
        )


# Display chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"- **{source['file']}** | Page: {source['page']}"
                    )


# Chat input
question = st.chat_input("Ask your learning question...")


if question:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # Build conversation history BEFORE adding current question
    history = "\n".join(
        st.session_state.conversation_history
    )


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = process_question(
                    question=question,
                    history=history
                )

                answer = result["answer"]
                category = result["category"]

                # Display answer
                st.markdown(answer)

                # Convert sources to readable format
                readable_sources = []

                for doc in result.get("sources", []):

                    source = doc.metadata.get(
                        "source", "Unknown"
                    )

                    source_name = source.replace(
                        "\\", "/"
                    ).split("/")[-1]

                    page = doc.metadata.get("page")

                    if isinstance(page, int):
                        page += 1

                    readable_sources.append({
                        "file": source_name,
                        "page": page
                    })


                # Display route
                st.caption(f"Route: {category}")


                # Display sources
                if readable_sources:

                    with st.expander("📚 View sources"):

                        for source in readable_sources:

                            st.markdown(
                                f"- **{source['file']}** | Page: {source['page']}"
                            )


                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": readable_sources,
                    "category": category
                })


                # Save conversation history
                st.session_state.conversation_history.append(
                    f"User: {question}"
                )

                st.session_state.conversation_history.append(
                    f"Assistant: {answer}"
                )


            except Exception as e:

                st.error(f"Error: {e}")
