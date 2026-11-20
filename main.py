from src.rag_pipeline import process_question


def main():
    print("=" * 60)
    print("       EdTech Learning & Career RAG Chatbot")
    print("=" * 60)
    print("Ask questions about your learning materials.")
    print("Type 'exit' or 'quit' to stop.\n")

    conversation_history = []

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! Happy learning.")
            break

        if not question:
            continue

        history = "\n".join(conversation_history)

        try:
            result = process_question(
                question=question,
                history=history
            )

            print(f"\nRoute: {result['category']}")
            print(f"\nChatbot: {result['answer']}\n")

            if result["category"] == "DOCUMENT" and result["sources"]:
                print("Sources:")
                for doc in result["sources"]:
                    source = doc.metadata.get("source", "Unknown")
                    source_name = source.replace("\\", "/").split("/")[-1]

                    page = doc.metadata.get("page")
                    if isinstance(page, int):
                        page = page + 1

                    print(f"- {source_name} | Page: {page}")
                print()

            conversation_history.append(f"User: {question}")
            conversation_history.append(
                f"Assistant: {result['answer']}"
            )

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()