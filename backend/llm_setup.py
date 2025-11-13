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

    # ✅ Handle both vectorstore and retriever cases safely
    if hasattr(vectorstore_or_retriever, "as_retriever"):
        retriever = vectorstore_or_retriever.as_retriever(search_kwargs={"k": 3})
    else:
        retriever = vectorstore_or_retriever  # already a retriever

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=500,
        api_key=os.environ["GROQ_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_template("""
You are FinSolve AI Assistant. You answer questions based only on the provided context.

Each user has a department role: {user_role}.
If the question seems unrelated to their department, politely refuse access.

---

Context:
{context}

---

Question: {question}

Answer concisely and professionally for the {user_role} department.
""")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    # ✅ Create chain parts
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    # ✅ Safe invoke to prevent pydantic_core validation issues
    def safe_invoke(inputs: dict):
        try:
            result = retrieval_chain.invoke(inputs)
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("result") or str(result)
                source_docs = result.get("context") or result.get("source_documents") or []
            else:
                answer, source_docs = str(result), []

            return {
                "answer": answer,
                "source_documents": source_docs
            }
        except Exception as e:
            return {"answer": f"⚠️ Internal Error: {str(e)}", "source_documents": []}

    retrieval_chain.invoke = safe_invoke
    print(f"✅ QA chain ready for role: {user_role}")
    return retrieval_chain
