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
# ============================================================
# LEITURA DOS DOCUMENTOS
# ============================================================

def carregar_documentos():

    documentos = []

    arquivos_pdf = [
        arquivo
        for arquivo in os.listdir(DATA_DIR)
        if arquivo.lower().endswith(".pdf")
    ]

    arquivos_pdf.sort()

    for arquivo in arquivos_pdf:

        caminho = os.path.join(DATA_DIR, arquivo)

        reader = PdfReader(caminho)

        for numero_pagina, pagina in enumerate(
            reader.pages,
            start=1
        ):

            texto = pagina.extract_text()

            if texto and texto.strip():

                documentos.append({
                    "fonte": arquivo,
                    "pagina": numero_pagina,
                    "texto": texto.strip()
                })

    return documentos


documentos = carregar_documentos()

print(
    f"✅ Documentos carregados: {len(documentos)} páginas com texto"
)
# ============================================================
# CHUNKING DOS DOCUMENTOS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = []

for documento in documentos:

    partes = text_splitter.split_text(
        documento["texto"]
    )

    for numero_chunk, texto_chunk in enumerate(
        partes,
        start=1
    ):

        chunks.append({
            "fonte": documento["fonte"],
            "pagina": documento["pagina"],
            "chunk": numero_chunk,
            "texto": texto_chunk
        })


print(f"✅ Chunks criados: {len(chunks)}")
