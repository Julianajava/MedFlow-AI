# ============================================================
# MEDFLOW AI
# Agente RAG para consulta de protocolos e documentos clínicos
# ============================================================

import os
import base64

import gradio as gr
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODELO_EMBEDDINGS = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MODELO_GEMINI = os.environ.get(
    "MODELO_GEMINI",
    "gemini-3.5-flash-lite"
)

NOME_COLLECTION = "medflow_protocolos"

print("🚀 Iniciando MedFlow AI...")


# ============================================================
# VERIFICAÇÕES INICIAIS
# ============================================================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"A pasta de documentos não foi encontrada: {DATA_DIR}"
    )

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "A variável de ambiente GEMINI_API_KEY não foi configurada."
    )


# ============================================================
# LEITURA DOS PDFs
# ============================================================

def carregar_documentos():
    documentos = []

    arquivos_pdf = [
        arquivo
        for arquivo in os.listdir(DATA_DIR)
        if arquivo.lower().endswith(".pdf")
    ]

    arquivos_pdf.sort()

    if not arquivos_pdf:
        raise FileNotFoundError(
            "Nenhum arquivo PDF foi encontrado na pasta data."
        )

    for arquivo in arquivos_pdf:
        caminho = os.path.join(DATA_DIR, arquivo)

        print(f"📄 Lendo: {arquivo}")

        reader = PdfReader(caminho)

        for numero_pagina, pagina in enumerate(
            reader.pages,
            start=1
        ):
            texto = pagina.extract_text()

            if texto and texto.strip():
                documentos.append(
                    {
                        "fonte": arquivo,
                        "pagina": numero_pagina,
                        "texto": texto.strip(),
                    }
                )

    return documentos


documentos = carregar_documentos()

print(
    f"✅ Documentos carregados: "
    f"{len(documentos)} páginas com texto"
)


# ============================================================
# CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""],
)

chunks = []

for documento in documentos:
    partes = text_splitter.split_text(
        documento["texto"]
    )

    for numero_chunk, parte in enumerate(
        partes,
        start=1
    ):
        chunks.append(
            {
                "fonte": documento["fonte"],
                "pagina": documento["pagina"],
                "chunk": numero_chunk,
                "texto": parte,
            }
        )

if not chunks:
    raise ValueError(
        "Nenhum chunk foi criado a partir dos documentos."
    )

print(f"✅ Chunks criados: {len(chunks)}")


# ============================================================
# EMBEDDINGS
# ============================================================

print("🧠 Carregando modelo de embeddings...")

modelo_embeddings = SentenceTransformer(
    MODELO_EMBEDDINGS
)

textos_chunks = [
    item["texto"]
    for item in chunks
]

embeddings = modelo_embeddings.encode(
    textos_chunks,
    normalize_embeddings=True,
    show_progress_bar=False,
)

print(
    f"✅ Embeddings criados: {len(embeddings)}"
)


# ============================================================
# CHROMADB
# ============================================================

client_chroma = chromadb.Client()

collection = client_chroma.get_or_create_collection(
    name=NOME_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# Evita duplicação caso a aplicação seja recarregada
if collection.count() == 0:
    ids = []
    documentos_chroma = []
    metadados = []
    vetores = []

    for i, item in enumerate(chunks):
        ids.append(f"chunk_{i}")

        documentos_chroma.append(
            item["texto"]
        )

        metadados.append(
            {
                "fonte": item["fonte"],
                "pagina": item["pagina"],
                "chunk": item["chunk"],
            }
        )

        vetores.append(
            embeddings[i].tolist()
        )

    collection.add(
        ids=ids,
        documents=documentos_chroma,
        metadatas=metadados,
        embeddings=vetores,
    )

print(
    f"✅ Banco vetorial pronto: "
    f"{collection.count()} chunks"
)


# ============================================================
# GEMINI
# ============================================================

client_gemini = genai.Client(
    api_key=GEMINI_API_KEY
)

print("✅ Gemini configurado")


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def buscar_contexto(pergunta, quantidade=1):
    embedding_pergunta = modelo_embeddings.encode(
        [pergunta],
        normalize_embeddings=True,
    )

    resultado = collection.query(
        query_embeddings=embedding_pergunta.tolist(),
        n_results=quantidade,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    contextos = []

    documentos_resultado = resultado["documents"][0]
    metadados_resultado = resultado["metadatas"][0]
    distancias_resultado = resultado["distances"][0]

    for i in range(len(documentos_resultado)):
        contextos.append(
            {
                "texto": documentos_resultado[i],
                "fonte": metadados_resultado[i]["fonte"],
                "pagina": metadados_resultado[i]["pagina"],
                "chunk": metadados_resultado[i]["chunk"],
                "similaridade": (
                    1 - distancias_resultado[i]
                ),
            }
        )

    return contextos


# ============================================================
# RESPOSTA RAG
# ============================================================

def responder_medflow(pergunta):
    if not pergunta:
        return (
            "⚠️ Digite uma pergunta para consultar "
            "a base de conhecimento."
        )

    pergunta = pergunta.strip()

    if not pergunta:
        return (
            "⚠️ Digite uma pergunta para consultar "
            "a base de conhecimento."
        )

    try:
        resultados = buscar_contexto(
            pergunta,
            quantidade=1,
        )

        if not resultados:
            return (
                "Não encontrei essa informação "
                "na base de conhecimento."
            )

        melhor = resultados[0]

        contexto = f"""
Fonte: {melhor["fonte"]}
Página: {melhor["pagina"]}

{melhor["texto"]}
"""

        prompt = f"""
Você é o MedFlow AI, um assistente inteligente para consulta
de protocolos e documentos clínicos.

Responda à pergunta utilizando SOMENTE as informações
presentes no CONTEXTO.

REGRAS:
- Não invente informações.
- Não utilize conhecimento externo ao contexto.
- Seja claro, objetivo e profissional.
- Não faça diagnóstico.
- Não faça prescrição.
- Não determine uma conduta clínica que não esteja descrita.
- Se a resposta não estiver no contexto, responda exatamente:
"Não encontrei essa informação na base de conhecimento."
- Quando houver resposta, informe a fonte e a página ao final.
- Não diga que algo é seguro ou indicado se o documento
  não afirmar isso.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

        resposta = client_gemini.models.generate_content(
            model=MODELO_GEMINI,
            contents=prompt,
        )

        if not resposta.text:
            return (
                "⚠️ Não foi possível gerar uma resposta. "
                "Tente novamente."
            )

        return resposta.text

    except Exception as erro:
        print(
            "❌ Erro durante a consulta:",
            repr(erro),
        )

        return (
            "### ❌ Não foi possível concluir a consulta\n\n"
            "Ocorreu um problema temporário durante "
            "o processamento. Tente novamente."
        )


# ============================================================
# IDENTIDADE VISUAL
# ============================================================

def carregar_logo():
    nomes_possiveis = [
        "MedFlow_AI_nova_identidade.png",
        "medflow_logo.png",
        "logo.png",
    ]

    for nome in nomes_possiveis:
        caminho = os.path.join(
            BASE_DIR,
            nome,
        )

        if os.path.isfile(caminho):
            with open(caminho, "rb") as imagem:
                return base64.b64encode(
                    imagem.read()
                ).decode("utf-8")

    return None


logo_base64 = carregar_logo()


# ============================================================
# CSS
# ============================================================

css_medflow = """
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background:
        radial-gradient(
            circle at top right,
            rgba(0, 194, 209, 0.10),
            transparent 35%
        ),
        linear-gradient(
            180deg,
            #f7fbfd 0%,
            #eef7fa 100%
        );
}

#medflow-main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 30px;
}

.hero-medflow {
    background:
        linear-gradient(
            135deg,
            #002f57 0%,
            #005f8f 55%,
            #00a7b5 100%
        );
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

.hero-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 35px;
    flex-wrap: wrap;
}

.hero-text {
    flex: 1;
    min-width: 280px;
}

.hero-logo {
    width: 270px;
    max-width: 100%;
    border-radius: 18px;
}

.info-card {
    background: white;
    border: 1px solid #dcecf2;
    border-radius: 18px;
    padding: 20px;
    min-height: 115px;
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
    border: 1px solid #f1e2a7;
    border-left: 5px solid #e4b534;
    padding: 15px;
    border-radius: 10px;
}

.footer-medflow {
    margin-top: 30px;
    padding: 20px 0;
    text-align: center;
    color: #718792;
    font-size: 12px;
}
"""


# ============================================================
# HERO
# ============================================================

if logo_base64:
    hero_html = f"""
    <div class="hero-medflow">
        <div class="hero-content">

            <div class="hero-text">
                <h1>MedFlow AI</h1>

                <p>
                    Assistente inteligente para consulta de
                    protocolos e documentos clínicos utilizando
                    tecnologia RAG.
                </p>

                <p>
                    <strong>
                        Conectando conhecimento,
                        melhorando decisões,
                        cuidando de vidas.
                    </strong>
                </p>
            </div>

            <div>
                <img
                    class="hero-logo"
                    src="data:image/png;base64,{logo_base64}"
                />
            </div>

        </div>
    </div>
    """
else:
    hero_html = """
    <div class="hero-medflow">

        <h1>MedFlow AI</h1>

        <p>
            Assistente inteligente para consulta de protocolos
            e documentos clínicos utilizando tecnologia RAG.
        </p>

        <p>
            <strong>
                Conectando conhecimento,
                melhorando decisões,
                cuidando de vidas.
            </strong>
        </p>

    </div>
    """


# ============================================================
# INTERFACE GRADIO
# ============================================================

with gr.Blocks(
    title="MedFlow AI"
) as app:

    gr.HTML(
        hero_html
    )

    with gr.Column(
        elem_id="medflow-main"
    ):

        with gr.Row():

            gr.HTML(
                """
                <div class="info-card">
                    <h3>📚 Base documental</h3>
                    <p>
                        Consulta informações diretamente
                        dos documentos disponíveis.
                    </p>
                </div>
                """
            )

            gr.HTML(
                """
                <div class="info-card">
                    <h3>🔎 Busca semântica</h3>
                    <p>
                        Localiza o trecho mais relevante
                        para cada pergunta.
                    </p>
                </div>
                """
            )

            gr.HTML(
                """
                <div class="info-card">
                    <h3>🧠 Tecnologia RAG</h3>
                    <p>
                        Respostas fundamentadas no contexto
                        recuperado da base documental.
                    </p>
                </div>
                """
            )

        with gr.Column(
            elem_id="consulta-card"
        ):

            gr.Markdown(
                "## Consulte o MedFlow AI"
            )

            gr.Markdown(
                "Faça uma pergunta relacionada "
                "à base de conhecimento."
            )

            pergunta = gr.Textbox(
                label="Pergunta",
                placeholder=(
                    "Ex.: O que fazer se o paciente "
                    "não cumpriu o jejum?"
                ),
                lines=3,
            )

            with gr.Row():

                botao = gr.Button(
                    "✦ Perguntar ao MedFlow AI",
                    variant="primary",
                )

                limpar = gr.Button(
                    "Limpar"
                )

            gr.Markdown(
                "### Resposta"
            )

            resposta = gr.Markdown(
                value=(
                    "*A resposta do MedFlow AI "
                    "aparecerá aqui.*"
                )
            )

            gr.Markdown(
                "### Perguntas para testar"
            )

            gr.Examples(
                examples=[
                    [
                        "O que fazer se o paciente "
                        "não cumpriu o jejum?"
                    ],
                    [
                        "O que fazer quando o paciente "
                        "relata alergia?"
                    ],
                    [
                        "Como proceder se houver "
                        "divergência de lateralidade?"
                    ],
                    [
                        "O que fazer quando um termo "
                        "obrigatório está pendente?"
                    ],
                    [
                        "O que fazer se o sistema "
                        "apresentar dados de outro paciente?"
                    ],
                    [
                        "Qual antibiótico deve ser "
                        "administrado antes da cirurgia?"
                    ],
                ],
                inputs=pergunta,
            )

        gr.HTML(
            """
            <div class="disclaimer">
                <strong>⚠️ Uso educacional:</strong>
                O MedFlow AI é um projeto acadêmico baseado
                em RAG. Não substitui avaliação profissional,
                prescrição, decisão clínica ou protocolos
                institucionais.
            </div>
            """
        )

        gr.HTML(
            """
            <div class="footer-medflow">
                <strong>MedFlow AI</strong><br>
                RAG • Inteligência Artificial • Saúde
            </div>
            """
        )

    botao.click(
        fn=responder_medflow,
        inputs=pergunta,
        outputs=resposta,
    )

    pergunta.submit(
        fn=responder_medflow,
        inputs=pergunta,
        outputs=resposta,
    )

    limpar.click(
        fn=lambda: (
            "",
            "*A resposta do MedFlow AI aparecerá aqui.*",
        ),
        inputs=[],
        outputs=[
            pergunta,
            resposta,
        ],
    )


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":

    porta = int(
        os.environ.get(
            "PORT",
            7860,
        )
    )

    print(
        f"🌐 Iniciando interface na porta {porta}"
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=porta,
        css=css_medflow,
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        show_error=True,
    )
