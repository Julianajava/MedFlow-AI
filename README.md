# MedFlow AI

Assistente inteligente baseado em **RAG — Retrieval-Augmented Generation** para consulta de protocolos e documentos clínicos.

> **Conectando conhecimento, melhorando decisões, cuidando de vidas.**

---

## Sobre o projeto

O **MedFlow AI** foi desenvolvido como projeto acadêmico para o Challenge Alura Agente.

A proposta é permitir que o usuário faça perguntas em linguagem natural e receba respostas baseadas exclusivamente em uma base documental previamente carregada.

O sistema utiliza técnicas de **RAG**, combinando busca semântica, embeddings, banco vetorial e um modelo generativo.

---

## Problema

Profissionais de saúde lidam diariamente com protocolos, orientações e documentos operacionais.

Localizar rapidamente uma informação específica pode exigir a consulta manual de diversos arquivos.

O MedFlow AI busca facilitar esse processo através de uma interface de perguntas e respostas.

---

## Solução

O usuário realiza uma pergunta.

O sistema:

1. Transforma a pergunta em um embedding.
2. Busca os trechos semanticamente mais próximos na base documental.
3. Recupera o contexto relevante.
4. Envia esse contexto para o modelo generativo.
5. Gera uma resposta baseada nos documentos.
6. Informa a fonte utilizada quando possível.

---

## Arquitetura

```text
Usuário
   ↓
Interface Gradio
   ↓
Pergunta
   ↓
Embedding da pergunta
   ↓
Busca semântica
   ↓
ChromaDB
   ↓
Trechos relevantes dos PDFs
   ↓
Contexto
   ↓
Prompt
   ↓
Gemini
   ↓
Resposta MedFlow AI
```

---

## Tecnologias utilizadas

* Python
* Google Colab
* LangChain Text Splitters
* Sentence Transformers
* Embeddings multilíngues
* ChromaDB
* Similaridade por cosseno
* Google Gemini API
* Gradio
* Git
* GitHub

---

## Modelo de embeddings

O projeto utiliza:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Os embeddings são normalizados e armazenados no ChromaDB utilizando similaridade por cosseno.

---

## Modelo generativo

O agente utiliza o modelo:

```text
gemini-3.5-flash-lite
```

A escolha foi feita buscando menor latência na geração das respostas.

---

## Base de conhecimento

A base documental do MedFlow AI é composta por cinco documentos:

```text
01_manual_admissao_preoperatorio_medflow_novo.pdf

02_protocolo_seguranca_paciente_medflow_novo.pdf

03_manual_documentacao_consentimentos_medflow_novo.pdf

04_guia_preparo_preoperatorio_medflow_novo.pdf

05_faq_operacional_medflow_novo.pdf
```

Os documentos foram desenvolvidos para fins acadêmicos com conteúdo compatível com práticas reconhecidas de segurança do paciente.

---

## Processamento dos documentos

O pipeline realiza:

```text
PDF
 ↓
Extração do texto
 ↓
Separação por páginas
 ↓
Chunking
 ↓
Embeddings
 ↓
Banco vetorial
```

Os documentos utilizados no projeto geraram:

```text
5 PDFs
10 páginas processadas
31 chunks
31 embeddings
384 dimensões por embedding
```

---

## Exemplos de perguntas

O agente pode responder perguntas como:

```text
O que fazer se o paciente não cumpriu o jejum?
```

```text
O que fazer quando o paciente relata alergia?
```

```text
Como proceder se houver divergência de lateralidade?
```

```text
O que fazer quando um termo obrigatório está pendente?
```

```text
O que fazer se o sistema apresentar dados de outro paciente?
```

---

## Exemplo de resposta

### Pergunta

```text
O que fazer se o paciente não cumpriu o jejum?
```

### Resposta

O MedFlow AI recupera a orientação presente na base documental e informa que deve ser registrado o que foi ingerido e o horário, quando possível, além de comunicar a equipe anestésica/cirúrgica.

A decisão de manter, adiar ou modificar o procedimento depende da avaliação profissional.

**Fonte:** FAQ Operacional — Pré-operatório e Segurança.

---

## Proteção contra alucinação

O agente foi configurado para não inventar informações.

Quando a resposta não está presente na base documental, o sistema responde:

```text
Não encontrei essa informação na base de conhecimento.
```

Exemplo:

```text
Qual antibiótico deve ser administrado antes da cirurgia?
```

Resposta:

```text
Não encontrei essa informação na base de conhecimento.
```

---

## Interface

O MedFlow AI possui uma interface criada com **Gradio**, contendo:

* Campo para perguntas
* Botão de consulta
* Área de resposta
* Exemplos de perguntas
* Identidade visual própria
* Aviso de uso educacional

---

## Como executar

O projeto foi desenvolvido inicialmente no Google Colab.

Para executar:

1. Abra o notebook `MedFlow_AI_RAG.ipynb`.
2. Instale as dependências.
3. Faça upload dos documentos PDF.
4. Configure sua chave da API do Gemini.
5. Execute as células na ordem.
6. Inicie a interface Gradio.

A chave da API não deve ser armazenada diretamente no código ou enviada ao GitHub.

---

## Estrutura prevista do repositório

```text
MedFlow-AI/
│
├── MedFlow_AI_RAG.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── 01_manual_admissao_preoperatorio_medflow_novo.pdf
│   ├── 02_protocolo_seguranca_paciente_medflow_novo.pdf
│   ├── 03_manual_documentacao_consentimentos_medflow_novo.pdf
│   ├── 04_guia_preparo_preoperatorio_medflow_novo.pdf
│   └── 05_faq_operacional_medflow_novo.pdf
│
└── screenshots/
    ├── interface_medflow.png
    └── deploy_oci.png
```

---

## Segurança e limitações

O MedFlow AI é um projeto acadêmico.

Ele não substitui:

* Avaliação médica
* Avaliação de enfermagem
* Prescrição
* Decisão clínica
* Protocolos institucionais
* Orientação de profissionais habilitados

As respostas devem ser interpretadas exclusivamente dentro do contexto educacional do projeto.

---

## Deploy

A aplicação será preparada para deploy na **Oracle Cloud Infrastructure — OCI**, conforme os requisitos do Challenge.

A evidência do deploy será adicionada ao repositório.

---

## Objetivo acadêmico

O projeto demonstra conceitos de:

* Inteligência Artificial
* Retrieval-Augmented Generation
* Embeddings
* Busca semântica
* Banco vetorial
* Engenharia de prompts
* Integração com LLM
* Desenvolvimento de agentes inteligentes

---

## Status

**Em desenvolvimento**

* [x] Documentação
* [x] Processamento de PDFs
* [x] Chunking
* [x] Embeddings
* [x] Banco vetorial
* [x] Busca semântica
* [x] Integração com Gemini
* [x] Agente RAG funcional
* [x] Interface Gradio
* [ ] Organização final do GitHub
* [ ] Deploy OCI

---

## MedFlow AI

**Conectando conhecimento, melhorando decisões, cuidando de vidas.**
