from vectorstore_setup import get_vectorstore_for_role


def main():
    role = "marketing"
    query = "What is the latest marketing budget?"

    retriever = get_vectorstore_for_role(role)

    # retriever.invoke returns a list of documents, not dict
    results = retriever.invoke(query)

    print(f"🔍 Retrieved {len(results)} documents for role: {role}")
    for doc in results:
        print(f"Document source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content snippet: {doc.page_content[:200]}...\n")


if __name__ == "__main__":
    main()
