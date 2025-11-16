# app.py
"""
Streamlit RAG Q&A — Gemini embeddings + ChromaDB + Gemini generation.

Requirements:
 - Set GEMINI_API_KEY as an environment variable / Streamlit secret.

How to run:
 1. pip install -r requirements.txt
 2. streamlit run app.py
"""
import os
import uuid
import textwrap
from typing import List, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import numpy as np

# Google GenAI SDK
from google import genai

# Chroma
import chromadb
from chromadb.config import Settings

# -------------------------
# Config
# -------------------------
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "rag_pages"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBEDDING_MODEL = "gemini-embedding-001"   # gemini embedding model (recommended)
GEN_MODEL = "gemini-2.5-flash"            # generation model

# -------------------------
# UI styling
# -------------------------
st.set_page_config(page_title="RAG URL Q&A — Streamlit + Gemini", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .main > .block-container { padding: 2rem 3rem; }
    .stApp { background: linear-gradient(180deg, #0b2545 0%, #07203a 100%); color: white; }
    .stMarkdown, .stText, .stButton>button, .stTextInput>div>input { color: #ffffff !important; }
    .card { background: white; color: #0b2545; padding: 1rem; border-radius: 12px; box-shadow: rgba(2,6,23,0.4) 0px 8px 24px; }
    .muted { color: #cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities
# -------------------------
def fetch_url_text(url: str, max_chars: int = 200_000) -> Tuple[str, str]:
    headers = {"User-Agent": "RAG-Streamlit-App/1.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    body = soup.find("main") or soup.find("article")
    if body:
        texts = body.get_text(separator="\n")
    else:
        ps = soup.find_all("p")
        if ps:
            texts = "\n\n".join(p.get_text() for p in ps)
        else:
            texts = soup.get_text(separator="\n")
    title_tag = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).netloc
    texts = texts.strip()
    if len(texts) > max_chars:
        texts = texts[:max_chars]
    return title_tag, texts

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(0, end - overlap)
    return chunks

# -------------------------
# GenAI client (singleton)
# -------------------------
@st.cache_resource(show_spinner=False)
def init_genai_client():
    # client will pick up GEMINI_API_KEY or GOOGLE_API_KEY environment variable
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # in Streamlit Cloud you should set this in Secrets
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable not set.")
    client = genai.Client(api_key=api_key)
    return client

# -------------------------
# Chroma client / collection
# -------------------------
@st.cache_resource(show_spinner=False)
def init_chroma(persist_directory: str = CHROMA_DIR):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    return client

@st.cache_resource(show_spinner=False)
def get_or_create_collection(chroma_client, name=CHROMA_COLLECTION_NAME):
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.create_collection(name)

# -------------------------
# Embedding helpers using GenAI
# -------------------------
def embed_texts(client: genai.Client, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Call Gemini embedding API for a list of texts (batching handled by SDK)."""
    # client.models.embed_content returns a response object; docs/examples show .embeddings or .embedding
    # We'll call client.models.embed_content and extract embeddings robustly.
    resp = client.models.embed_content(model=model, contents=texts)
    # response shape may contain resp.embeddings or resp.embedding; handle both
    embeddings = []
    if hasattr(resp, "embeddings"):
        for e in resp.embeddings:
            # some responses wrap embedding in .embedding
            if hasattr(e, "embedding"):
                embeddings.append(list(e.embedding))
            else:
                embeddings.append(list(e))
    elif hasattr(resp, "embedding"):
        # single embedding -> make list
        embeddings.append(list(resp.embedding))
    else:
        # fallback: try dict access
        j = resp.__dict__ if hasattr(resp, "__dict__") else dict(resp)
        if "embeddings" in j:
            for e in j["embeddings"]:
                embeddings.append(list(e.get("embedding") if isinstance(e, dict) else e))
        else:
            raise RuntimeError("Unexpected embedding response structure from GenAI SDK.")
    return embeddings

def embed_single(client: genai.Client, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    return embed_texts(client, [text], model=model)[0]

# -------------------------
# Chroma helpers (add & retrieve)
# -------------------------
def upsert_page_to_chroma(collection, url: str, title: str, chunks: List[str], embeddings: List[List[float]]):
    ids = [f"{uuid.uuid4()}_{i}" for i in range(len(chunks))]
    metadatas = [{"url": url, "title": title, "chunk_index": i} for i in range(len(chunks))]
    documents = [textwrap.shorten(c, width=2000, placeholder="") for c in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

def retrieve_relevant_chunks(collection, query_embedding: List[float], top_k: int = TOP_K):
    results = collection.query(query_embeddings=[query_embedding], n=top_k, include=["documents", "metadatas", "distances"])
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    hits = [{"doc": d, "meta": m, "dist": dist} for d, m, dist in zip(docs, metas, dists)]
    return hits

# -------------------------
# Gemini generation
# -------------------------
def genai_generate(client: genai.Client, prompt: str, model: str = GEN_MODEL, max_tokens: int = 800, temperature: float = 0.0) -> str:
    # Use client.models.generate_content
    resp = client.models.generate_content(model=model, contents=prompt, max_output_tokens=max_tokens, temperature=temperature)
    # resp.text is a convenience property from docs
    text = getattr(resp, "text", None)
    if text:
        return text
    # fallback: try extracting from parts
    if hasattr(resp, "parts") and len(resp.parts) > 0:
        parts_text = []
        for p in resp.parts:
            if hasattr(p, "text"):
                parts_text.append(p.text)
            elif hasattr(p, "content"):
                parts_text.append(getattr(p, "content", ""))
        return "\n".join(parts_text)
    # last fallback
    return str(resp)

# -------------------------
# Prompt builder
# -------------------------
def build_rag_prompt(question: str, retrieved: List[dict]) -> str:
    context_blocks = []
    for i, hit in enumerate(retrieved):
        meta = hit.get("meta", {})
        url = meta.get("url", "unknown")
        title = meta.get("title", "")
        doc = hit.get("doc", "")
        source_tag = f"[source:{i+1}]"
        context_blocks.append(f"{source_tag} TITLE: {title}\nURL: {url}\nCONTENT:\n{doc}\n")
    context_text = "\n---\n".join(context_blocks)
    prompt = (
        "You are an expert assistant. Use ONLY the information in the provided CONTEXT to answer the user's question. "
        "If the context does not contain an answer, say so and avoid hallucinations. Provide a clear, well-structured answer, "
        "followed by a short 'Sources' section listing which sources you used (by source tag) and the URL.\n\n"
        f"QUESTION: {question}\n\n"
        "CONTEXT:\n"
        f"{context_text}\n\n"
        "INSTRUCTIONS:\n"
        " - Provide a full explanation, step-by-step if helpful.\n"
        " - After the explanation, include a 'Sources' section that maps source tags to URLs.\n"
        " - Keep the answer factual and cite the exact source tags used.\n"
    )
    return prompt

# -------------------------
# Streamlit layout
# -------------------------
st.title("RAG URL Q&A — Streamlit + Gemini")
st.write("<div class='muted'>Paste a URL, ask a question about that page, and get a full explanation with citations.</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    url_input = st.text_input("Page URL", placeholder="https://example.com/article-or-doc", key="url_input")
    load_btn = st.button("Fetch & Index URL", key="index_btn")
    status_area = st.empty()

with col2:
    user_question = st.text_area("Your question about the page", height=140, placeholder="e.g., Summarize the main argument and list supporting evidence.")
    answer_btn = st.button("Ask AI", key="ask_btn")

# Initialize clients
try:
    genai_client = init_genai_client()
except Exception as e:
    st.error(f"GenAI client initialization error: {e}")
    st.stop()

chroma_client = init_chroma(CHROMA_DIR)
collection = get_or_create_collection(chroma_client, CHROMA_COLLECTION_NAME)

# Indexing flow
if load_btn:
    if not url_input:
        st.warning("Please enter a URL first.")
    else:
        status_area.info("Fetching URL...")
        try:
            title, page_text = fetch_url_text(url_input)
            status_area.success(f"Fetched: {title[:120]}")
            chunks = chunk_text(page_text)
            status_area.info(f"Text split into {len(chunks)} chunks. Generating embeddings (Gemini)...")
            embeddings = embed_texts(genai_client, chunks, model=EMBEDDING_MODEL)
            status_area.info("Upserting into ChromaDB...")
            upsert_page_to_chroma(collection, url_input, title, chunks, embeddings)
            # persist
            chroma_client.persist()
            status_area.success("Indexed and persisted to ChromaDB ✅")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error fetching or indexing URL: {e}")

# Asking flow
if answer_btn:
    if not user_question or not url_input:
        st.warning("Provide both a URL (indexed) and a question.")
    else:
        with st.spinner("Embedding question & retrieving relevant pieces..."):
            try:
                q_emb = embed_single(genai_client, user_question, model=EMBEDDING_MODEL)
                hits = retrieve_relevant_chunks(collection, q_emb, top_k=TOP_K)
            except Exception as e:
                st.error(f"Embedding/query error: {e}")
                hits = []

            if not hits or all(h["doc"] == "" for h in hits):
                st.info("No relevant content found for that URL in the vectorstore. Try indexing the page first.")
            else:
                prompt = build_rag_prompt(user_question, hits)
                st.markdown("### Retrieved context")
                for i, h in enumerate(hits):
                    meta = h["meta"]
                    st.write(f"**Source [{i+1}]** — {meta.get('title','')}")
                    st.write(f"<div class='card'>{h['doc'][:1000]}{'...' if len(h['doc'])>1000 else ''}</div>", unsafe_allow_html=True)
                st.markdown("---")
                st.info("Calling Gemini for answer generation...")
                try:
                    answer_text = genai_generate(genai_client, prompt, model=GEN_MODEL, max_tokens=800, temperature=0.0)
                except Exception as e:
                    st.error(f"Error calling Gemini: {e}")
                    answer_text = None

                if answer_text:
                    st.markdown("## AI Answer")
                    st.write(answer_text)
                    st.markdown("### Source mapping (from retrieved hits)")
                    for i, h in enumerate(hits):
                        meta = h["meta"]
                        st.markdown(f"- [source:{i+1}] → {meta.get('title','')} — {meta.get('url')}")
