# backend/ask_cli.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ✅ Suppress tokenizer fork warning

from llm_setup import get_qa_chain

def main():
    print("👋 Welcome to the Role-Based QA Assistant")
    user_role = input("🔐 Enter your role (e.g., finance, hr, engineering): ").strip().lower()

    qa_chain = get_qa_chain(user_role)

    print("\n💬 Ask any question related to your role documents (type 'exit' to quit):")
    while True:
        query = input("\n🧠 You: ")
        if query.lower() in {"exit", "quit"}:
            break

        result = qa_chain.invoke(query)

        print(f"\n📄 Retrieved {len(result['source_documents'])} matching document(s)")
        print("\n🤖 Answer:\n", result['result'])

        print("\n📚 Sources:")
        for doc in result['source_documents']:
            metadata = doc.metadata
            print(f"- {metadata.get('source', 'Unknown')} ({metadata.get('role', 'Unknown')})")

if __name__ == "__main__":
    main()
