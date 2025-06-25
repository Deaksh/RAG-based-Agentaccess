import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load env vars
load_dotenv()

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not cred_path or not os.path.exists(cred_path):
    raise FileNotFoundError(f"Invalid or missing path to service account: {cred_path}")

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()
