"""
API keys - Copy this file to config.py and add your actual keys.
config.py is in .gitignore so your keys are never committed.
"""
import os

OPENAI_API_KEY = "sk-proj-YOUR_OPENAI_KEY_HERE"
COHERE_API_KEY = "YOUR_COHERE_KEY_HERE"
LANGCHAIN_API_KEY = ""  # optional, for LangSmith tracing

# Set env so all modules use these keys (no need to edit elsewhere)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"




