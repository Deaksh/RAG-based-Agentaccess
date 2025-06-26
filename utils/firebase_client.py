import base64
import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    encoded_key = st.secrets["firebase"]["encoded_key"]
    decoded_key_json = base64.b64decode(encoded_key).decode("utf-8")
    service_account_info = json.loads(decoded_key_json)
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_chat_history(user_email, chat_history):
    try:
        doc_ref = db.collection("chat_histories").document(user_email)
        doc_ref.set({"history": chat_history})
    except Exception as e:
        import traceback
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
