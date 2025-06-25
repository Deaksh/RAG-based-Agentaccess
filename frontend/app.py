# frontend/app.py

import os
import sys
import streamlit as st

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from backend.llm_setup import get_qa_chain
from utils.auth import authenticate_user, register_user, list_users

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
                st.session_state.chat_history = []  # 🚨 Auto-clear previous history
                st.success(f"✅ Logged in as {role.upper()}")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

# Admin Console
elif st.session_state.role == "admin":
    st.success(f"👑 Admin logged in: {st.session_state.email}")
    st.header("🔧 Admin Dashboard")
    st.write("Use this panel to register new users and assign them roles.")

    with st.form("admin_register_form"):
        new_email = st.text_input("New User Email")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Assign Role", [
            "finance", "marketing", "hr", "engineering", "c-level", "employee"
        ])
        submitted = st.form_submit_button("Register User")

        if submitted:
            if register_user(new_email.strip().lower(), new_password, new_role):
                st.success(f"✅ User '{new_email}' registered as '{new_role}'")
            else:
                st.warning("⚠️ User already exists.")

    st.markdown("---")
    st.subheader("📋 Registered Users")

    for user in list_users():
        st.markdown(f"- **{user['email']}** ({user['role'].upper()})")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# Regular Chat UI
else:
    st.success(f"🔐 Logged in as: {st.session_state.email} | Role: {st.session_state.role.upper()}")
    st.write("➡️ Ask department-specific questions below.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask your question")

    if user_question:
        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

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

        access_denied_phrases = [
            "you do not have permission",
            "this question seems related to a different department"
        ]

        if not any(phrase in answer.lower() for phrase in access_denied_phrases):
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

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
