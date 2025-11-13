# backend/llm_setup.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from vectorstore_setup import get_vectorstore_for_role
import os


def get_qa_chain(user_role: str):
    retriever = get_vectorstore_for_role(user_role)

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=500,
        api_key=os.environ["GROQ_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_template("""
You are an expert assistant helping users answer questions about internal role-based documents.

Use ONLY the provided context below to answer the question. Do not use any external knowledge.

If the answer cannot be found in the context, assume the user may be asking about another department’s documents.
In that case, respond with:

"You do not have permission to access this information.
🔒 Your role: {user_role}
This question seems related to a different department.
👉 Please rephrase your question within the scope of your department."

Try to:
- Be concise and factual.
- Format the answer in bullet points or short paragraphs.
- Mention the role and filename when relevant, using metadata if available.

---

Context:
{context}

---

Question: {question}

Answer:
""")

    # ✅ Core QA chain that expects `context` + `question` + `user_role`
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    # ✅ Retrieval chain that expects input as `input`
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    # ✅ Final bridging chain (maps user input → retriever input → prompt variables)
    full_chain = (
        {
            "input": lambda x: x["question"],  # retriever wants `input`
            "question": lambda x: x["question"],  # prompt wants `question`
            "context": lambda x: x.get("context", ""),
            "user_role": lambda _: user_role,
        }
        | rag_chain
        | StrOutputParser()
    )

    print("✅ QA chain ready with expected keys:", prompt.input_variables)
    return full_chain
