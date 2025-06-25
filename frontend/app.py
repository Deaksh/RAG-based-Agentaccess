# frontend/app.py

import os
import sys
import streamlit as st

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from backend.llm_setup import get_qa_chain
from utils.auth import authenticate_user

import streamlit as st
from qdrant_client import QdrantClient
import os

st.title("🛠️ Qdrant Collection Debugger")

QDRANT_COLLECTION = "finrolebot"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

st.write("### Collection info:")
collection_info = client.get_collection(QDRANT_COLLECTION)
st.json(collection_info.model_dump())

st.write("### Sample points from collection:")
try:
    points = client.scroll(collection_name=QDRANT_COLLECTION, limit=5)
    for point in points.points:
        st.write(f"Point ID: {point.id}")
        st.json(point.payload)  # This contains your metadata
except Exception as e:
    st.error(f"Error fetching points: {e}")

st.write("### Check if 'metadata.role' field exists in payloads")
try:
    points = client.scroll(collection_name=QDRANT_COLLECTION, limit=10)
    roles = [point.payload.get("role") or point.payload.get("metadata", {}).get("role") for point in points.points]
    st.write(roles)
except Exception as e:
    st.error(f"Error reading roles: {e}")


st.set_page_config(page_title="FinSolve Assistant", page_icon="💼", layout="wide")
st.title("💼 FinSolve Role-Based Assistant")
st.subheader("Login to access your department-specific insights")

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Login flow
if not st.session_state.logged_in:
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            role = authenticate_user(email, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.session_state.role = role.strip().lower()
                st.success(f"✅ Logged in as {role.upper()}")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

# Chat UI
else:
    st.success(f"🔐 Logged in as: {st.session_state.email} | Role: {st.session_state.role.upper()}")
    st.write("➡️ Ask department-specific questions below.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input for next question
    user_question = st.chat_input("Ask your question")

    if user_question:
        # Append user question
        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Load role-based QA chain
        qa_chain = get_qa_chain(st.session_state.role)

        with st.spinner("🔍 Searching your documents..."):
            response = qa_chain.invoke({
                "question": user_question,
                "user_role": st.session_state.role,
                "chat_history": st.session_state.chat_history
            })

        answer = response["answer"]
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # 🚫 Suppress sources if role-access violation is returned
        access_denied_phrases = [
            "you do not have permission",
            "this question seems related to a different department"
        ]

        if not any(phrase in response["answer"].lower() for phrase in access_denied_phrases):
            st.markdown("### 📚 Sources")
            for i, doc in enumerate(response["source_documents"]):
                metadata = doc.metadata
                source = metadata.get("source", "Unknown")
                role = metadata.get("role", "Unknown")
                content = doc.page_content.strip()
                st.markdown(f"- **Filename**: {source}")
                st.markdown(f"  **Role**: {role}")
                with st.expander(f"{i + 1}. {source} ({role})"):
                    st.markdown(f"> {content[:500]}...")
        else:
            st.info("🔒 No document sources shown due to access restrictions.")

    # Optional: Clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
