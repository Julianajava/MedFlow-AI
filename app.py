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
)# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def buscar_contexto(pergunta, quantidade=1):

    embedding_pergunta = modelo_embeddings.encode(
        [pergunta],
        normalize_embeddings=True
    )

    resultado = collection.query(
        query_embeddings=embedding_pergunta.tolist(),
        n_results=quantidade,
        include=["documents", "metadatas", "distances"]
    )

    contextos = []

    for i in range(len(resultado["documents"][0])):

        contextos.append({
            "texto": resultado["documents"][0][i],
            "fonte": resultado["metadatas"][0][i]["fonte"],
            "pagina": resultado["metadatas"][0][i]["pagina"],
            "chunk": resultado["metadatas"][0][i]["chunk"],
            "similaridade": 1 - resultado["distances"][0][i]
        })

    return contextos


print("✅ Função de busca semântica criada")
# ============================================================
# CONEXÃO COM O GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "A variável de ambiente GEMINI_API_KEY não foi configurada."
    )

client_gemini = genai.Client(
    api_key=GEMINI_API_KEY
)

print("✅ Cliente Gemini configurado com sucesso")
# ============================================================
# GERAÇÃO DA RESPOSTA
# ============================================================

def responder_medflow(pergunta):

    if not pergunta or not pergunta.strip():
        return "Digite uma pergunta para consultar a base de conhecimento."

    # Busca o trecho mais relevante
    resultados = buscar_contexto(
        pergunta,
        quantidade=1
    )

    melhor_resultado = resultados[0]

    contexto = f"""
Fonte: {melhor_resultado['fonte']}
Página: {melhor_resultado['pagina']}

{melhor_resultado['texto']}
"""

    prompt = f"""
Você é o MedFlow AI, um assistente inteligente para consulta
de protocolos e documentos clínicos.

Responda à pergunta utilizando SOMENTE o contexto fornecido.

Regras:
- Não invente informações.
- Não utilize conhecimento externo ao contexto.
- Seja claro, objetivo e profissional.
- Não faça diagnóstico ou prescrição.
- Se a resposta não estiver disponível no contexto, responda:
"Não encontrei essa informação na base de conhecimento."
- Informe a fonte utilizada ao final.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

    resposta = client_gemini.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt
    )

    if not resposta.text:
        return "Não foi possível gerar uma resposta."

    return resposta.text


print("✅ Função de resposta do MedFlow AI criada")
# ============================================================
# INTERFACE GRADIO
# ============================================================

css_medflow = """
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background:
        radial-gradient(circle at top right, rgba(0, 194, 209, 0.10), transparent 35%),
        linear-gradient(180deg, #f7fbfd 0%, #eef7fa 100%);
}

#medflow-main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 30px;
}

.hero-medflow {
    background: linear-gradient(135deg, #002f57 0%, #005f8f 55%, #00a7b5 100%);
    color: white;
    padding: 45px;
    border-radius: 0 0 28px 28px;
    margin-bottom: 25px;
}

.hero-medflow h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 10px;
}

.info-card {
    background: white;
    border: 1px solid #dcecf2;
    border-radius: 18px;
    padding: 20px;
    min-height: 110px;
}

#consulta-card {
    background: white;
    border-radius: 22px;
    padding: 25px !important;
    margin-top: 20px;
    border: 1px solid #d8eaf0;
}

.disclaimer {
    margin-top: 24px;
    background: #fffdf5;
    border-left: 5px solid #e4b534;
    padding: 15px;
    border-radius: 10px;
}

.footer-medflow {
    margin-top: 30px;
    text-align: center;
    color: #718792;
    font-size: 12px;
}
"""

with gr.Blocks(title="MedFlow AI") as app:

    gr.HTML("""
        <div class="hero-medflow">
            <h1>MedFlow AI</h1>

            <p>
                Assistente inteligente para consulta de protocolos
                e documentos clínicos utilizando tecnologia RAG.
            </p>

            <p>
                <strong>
                Conectando conhecimento, melhorando decisões,
                cuidando de vidas.
                </strong>
            </p>
        </div>
    """)

    with gr.Column(elem_id="medflow-main"):

        with gr.Row():

            gr.HTML("""
                <div class="info-card">
                    <h3>📚 Base documental</h3>
                    <p>
                        Consulta informações diretamente dos
                        documentos disponíveis.
                    </p>
                </div>
            """)

            gr.HTML("""
                <div class="info-card">
                    <h3>🔎 Busca semântica</h3>
                    <p>
                        Localiza o trecho mais relevante
                        para cada pergunta.
                    </p>
                </div>
            """)

            gr.HTML("""
                <div class="info-card">
                    <h3>🧠 Tecnologia RAG</h3>
                    <p>
                        Respostas fundamentadas no contexto
                        recuperado da base documental.
                    </p>
                </div>
            """)

        with gr.Column(elem_id="consulta-card"):

            gr.Markdown("## Consulte o MedFlow AI")

            pergunta = gr.Textbox(
                label="Pergunta",
                placeholder="Ex.: O que fazer se o paciente não cumpriu o jejum?",
                lines=3
            )

            with gr.Row():

                botao = gr.Button(
                    "Perguntar ao MedFlow AI",
                    variant="primary"
                )

                limpar = gr.Button(
                    "Limpar"
                )

            resposta = gr.Markdown(
                value="*A resposta aparecerá aqui.*"
            )

            gr.Examples(
                examples=[
                    ["O que fazer se o paciente não cumpriu o jejum?"],
                    ["O que fazer quando o paciente relata alergia?"],
                    ["Como proceder se houver divergência de lateralidade?"],
                    ["O que fazer quando um termo obrigatório está pendente?"],
                    ["O que fazer se o sistema apresentar dados de outro paciente?"]
                ],
                inputs=pergunta
            )

        gr.HTML("""
            <div class="disclaimer">
                <strong>⚠️ Uso educacional:</strong>
                O MedFlow AI é um projeto acadêmico baseado em RAG.
                Não substitui avaliação profissional, prescrição,
                decisão clínica ou protocolos institucionais.
            </div>
        """)

        gr.HTML("""
            <div class="footer-medflow">
                <strong>MedFlow AI</strong><br>
                RAG • Inteligência Artificial • Saúde
            </div>
        """)

    botao.click(
        fn=responder_medflow,
        inputs=pergunta,
        outputs=resposta
    )

    pergunta.submit(
        fn=responder_medflow,
        inputs=pergunta,
        outputs=resposta
    )

    limpar.click(
        fn=lambda: ("", "*A resposta aparecerá aqui.*"),
        inputs=[],
        outputs=[pergunta, resposta]
    )
