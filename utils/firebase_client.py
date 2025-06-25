# utils/firebase_client.py

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app only once
if not firebase_admin._apps:
    # Load Firebase credentials from Streamlit secrets
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

def save_chat_history(user_email, chat_history):
    doc_ref = db.collection("chat_histories").document(user_email)
    doc_ref.set({"history": chat_history})

def load_chat_history(user_email):
    doc_ref = db.collection("chat_histories").document(user_email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("history", [])
    return []
