# utils/auth.py

import json
import os

USER_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# Initialize file with default users if it doesn't exist
DEFAULT_USERS = {
    "alice@finsolve.com": {"password": "finance123", "role": "finance"},
    "bob@finsolve.com": {"password": "market123", "role": "marketing"},
    "carol@finsolve.com": {"password": "hr123", "role": "hr"},
    "dave@finsolve.com": {"password": "eng123", "role": "engineering"},
    "ceo@finsolve.com": {"password": "ceo123", "role": "c-level"},
    "employee@finsolve.com": {"password": "emp123", "role": "employee"},
    "admin@finsolve.com": {"password": "admin123", "role": "admin"}
}

def _load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump(DEFAULT_USERS, f, indent=2)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def _save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

def authenticate_user(email, password):
    users = _load_users()
    user = users.get(email)
    if user and user["password"] == password:
        return user["role"]
    return None

def register_user(email, password, role):
    users = _load_users()
    if email in users:
        return False  # Already exists
    users[email] = {"password": password, "role": role}
    _save_users(users)
    return True

def list_users():
    users = _load_users()
    return [{"email": email, "role": info["role"]} for email, info in users.items()]
