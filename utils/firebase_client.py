import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred = credentials.Certificate('utils/rag-based-assistant-4fc06-firebase-adminsdk-fbsvc-f9a4b89839.json')  # Your downloaded Firebase credentials JSON
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_chat_history(email, chat_history):
    doc_ref = db.collection('chat_histories').document(email)
    doc_ref.set({'history': chat_history})

def load_chat_history(email):
    doc_ref = db.collection('chat_histories').document(email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get('history', [])
    return []
