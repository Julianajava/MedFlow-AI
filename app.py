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
# ============================================================
# EMBEDDINGS E BANCO VETORIAL
# ============================================================

modelo_embeddings = SentenceTransformer(
    MODELO_EMBEDDINGS
)

textos_chunks = [
    item["texto"]
    for item in chunks
]

embeddings = modelo_embeddings.encode(
    textos_chunks,
    normalize_embeddings=True
)

client_chroma = chromadb.Client()

collection = client_chroma.get_or_create_collection(
    name="medflow_protocolos",
    metadata={"hnsw:space": "cosine"}
)

ids = []
documentos_chroma = []
metadados = []
vetores = []

for i, item in enumerate(chunks):

    ids.append(
        f"chunk_{i}"
    )

    documentos_chroma.append(
        item["texto"]
    )

    metadados.append({
        "fonte": item["fonte"],
        "pagina": item["pagina"],
        "chunk": item["chunk"]
    })

    vetores.append(
        embeddings[i].tolist()
    )

# Só adiciona se a coleção estiver vazia
if collection.count() == 0:

    collection.add(
        ids=ids,
        documents=documentos_chroma,
        metadatas=metadados,
        embeddings=vetores
    )

print(
    f"✅ Banco vetorial pronto: {collection.count()} chunks"
)
