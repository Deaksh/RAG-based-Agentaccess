# utils/firebase_client.py

import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not cred_path or not os.path.exists(cred_path):
    raise FileNotFoundError(f"Missing or invalid path to service account: {cred_path}")

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

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
