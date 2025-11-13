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

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

def get_qa_chain(user_role: str):
    """
    Returns a retrieval-based QA chain tailored to the given user role.
    """
    vectorstore = get_vectorstore_for_role(user_role)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are FinSolve AI Assistant. Your job is to answer user questions based only on the provided context.
Each user has a specific department role: {user_role}.
If the question is outside their department's access, politely refuse.

Context:
{context}

Question: {question}

Answer concisely and professionally for the {user_role} department.
""")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    # ✅ Wrap in safe invoke to prevent pydantic_core validation errors
    def safe_invoke(inputs: dict):
        try:
            result = retrieval_chain.invoke(inputs)
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("result") or str(result)
            else:
                answer = str(result)

            return {
                "answer": answer,
                "source_documents": result.get("context", []) if isinstance(result, dict) else []
            }
        except Exception as e:
            return {"answer": f"⚠️ Internal Error: {str(e)}", "source_documents": []}

    retrieval_chain.invoke = safe_invoke
    return retrieval_chain
