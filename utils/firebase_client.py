# utils/firebase_client.py
import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    service_account_info = {k: v for k, v in st.secrets["firebase"].items()}
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

db = firestore.client()



def save_chat_history(user_email, chat_history):
    try:
        doc_ref = db.collection("chat_histories").document(user_email)
        doc_ref.set({"history": chat_history})
    except Exception as e:
        import traceback
        import streamlit as st
        st.error("🔥 Firestore write failed.")
        st.text("Exception details:")
        st.text(str(e))
        st.text("Traceback:")
        st.text(traceback.format_exc())


def load_chat_history(user_email):
    doc_ref = db.collection("chat_histories").document(user_email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("history", [])
    return []
