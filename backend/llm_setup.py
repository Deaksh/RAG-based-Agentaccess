from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from vectorstore_setup import get_vectorstore_for_role
from dotenv import load_dotenv
import os

load_dotenv()


def get_qa_chain(user_role: str):
    """
    Returns a retrieval-based QA chain for the given user role.
    """
    vectorstore_or_retriever = get_vectorstore_for_role(user_role)

    # ✅ Handle both vectorstore and retriever cases
    if hasattr(vectorstore_or_retriever, "as_retriever"):
        retriever = vectorstore_or_retriever.as_retriever(search_kwargs={"k": 3})
    else:
        retriever = vectorstore_or_retriever  # already a retriever

    # ✅ LLM setup
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=500,
        api_key=os.environ["GROQ_API_KEY"],
    )

    # ✅ Updated prompt: uses {input} instead of {question}
    prompt = ChatPromptTemplate.from_template("""
You are FinSolve AI Assistant. You answer questions based only on the provided context.

Each user has a department role: {user_role}.
If the question seems unrelated to their department, politely refuse access.

---

Context:
{context}

---

Question: {input}

Answer concisely and professionally for the {user_role} department.
""")

    # ✅ Conversation memory (align with "input" and "answer")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="answer",
    )

    # ✅ Document chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    # ✅ Retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    # ✅ Safe wrapper for robust output
    class SafeQAWrapper:
        def __init__(self, chain):
            self.chain = chain

        def invoke(self, inputs: dict):
            try:
                print("🧩 Invoked with:", inputs)
                result = self.chain.invoke(inputs)
                print("✅ Chain result:", result)

                # Extract answer and sources
                if isinstance(result, dict):
                    answer = result.get("answer") or result.get("result") or str(result)
                    source_docs = result.get("source_documents") or []
                else:
                    answer, source_docs = str(result), []
                return {"answer": answer, "source_documents": source_docs}
            except Exception as e:
                print("❌ Internal error:", e)
                return {"answer": f"⚠️ Internal Error: {str(e)}", "source_documents": []}

    print(f"✅ QA chain ready for role: {user_role}")
    return SafeQAWrapper(retrieval_chain)
