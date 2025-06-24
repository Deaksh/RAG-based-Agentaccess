# utils/auth.py

USER_CREDENTIALS = {
    "alice@finsolve.com": {"password": "finance123", "role": "finance"},
    "bob@finsolve.com": {"password": "market123", "role": "marketing"},
    "carol@finsolve.com": {"password": "hr123", "role": "hr"},
    "dave@finsolve.com": {"password": "eng123", "role": "engineering"},
    "ceo@finsolve.com": {"password": "ceo123", "role": "c-level"},
    "employee@finsolve.com": {"password": "emp123", "role": "employee"},
}

def authenticate_user(email, password):
    user = USER_CREDENTIALS.get(email)
    if user and user["password"] == password:
        return user["role"]
    return None
