# ============================================================
# MEDFLOW AI
# Aplicação principal
# ============================================================

import os

import gradio as gr
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATA_DIR = "data"

MODELO_EMBEDDINGS = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MODELO_GEMINI = "gemini-3.5-flash-lite"

print("✅ MedFlow AI iniciado")
