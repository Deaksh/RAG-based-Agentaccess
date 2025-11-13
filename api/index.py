# api/index.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import subprocess
import os

app = FastAPI()

@app.get("/")
def home():
    """Redirect to the Streamlit frontend."""
    return RedirectResponse(url="/streamlit")

@app.get("/streamlit")
def run_streamlit():
    """Launch the Streamlit app."""
    streamlit_file = os.path.join("frontend", "app.py")

    # Run Streamlit as a subprocess
    subprocess.Popen(["streamlit", "run", streamlit_file, "--server.port", "7860", "--server.address", "0.0.0.0"])
    return {"status": "Streamlit launched on port 7860"}
