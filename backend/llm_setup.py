from langchain_groq import ChatGroq
# from langchain.chains import create_retrieval_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from backend.vectorstore_setup import get_vectorstore_for_role
from langchain.memory import ConversationBufferMemory
import os


def get_qa_chain(user_role: str):
    retriever = get_vectorstore_for_role(user_role)
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=500,
        api_key=os.environ["GROQ_API_KEY"],
    )

    prompt = PromptTemplate(
        input_variables=["context", "question", "user_role"],
        template="""
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
"""
    )

    # Memory is optional with the new retrieval chain pattern
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    # Create a chain that stuffs retrieved documents into the prompt
    combine_chain = create_stuff_documents_chain(llm, prompt)

    # Final retrieval-QA chain
    qa_chain = create_retrieval_chain(retriever, combine_chain)

    return qa_chain
