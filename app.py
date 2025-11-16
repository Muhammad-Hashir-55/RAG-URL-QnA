# app.py
"""
Streamlit RAG Q&A with URL input, sentence-transformers embeddings (all-mpnet-base-v2),
ChromaDB vectorstore, and Gemini 2.5-Flash for generation.

Environment variables required:
 - GEMINI_API_KEY  (your Gemini / Google Generative AI key)

How to run locally:
 1. pip install -r requirements.txt
 2. streamlit run app.py
"""
import os
import time
import uuid
import requests
import textwrap
from typing import List, Tuple

import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# -------------------------
# Configuration / Defaults
# -------------------------
MODEL_NAME = "thenlper/gte-small"
CHROMA_DIR = "chroma_db"  # persistent folder for chroma
CHROMA_COLLECTION_NAME = "rag_pages"
CHUNK_SIZE = 1000           # characters per chunk
CHUNK_OVERLAP = 200         # overlap between chunks
TOP_K = 4                   # number of retrieved chunks to pass to LLM
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_ENDPOINT = "https://api.generativeai.google/v1beta/models/{model}:generate"

# -------------------------
# Helpers: UI styling
# -------------------------
st.set_page_config(page_title="RAG URL Q&A — Streamlit + Gemini", page_icon="🤖", layout="wide")

# inject small CSS for deep-blue / white theme
st.markdown(
    """
    <style>
    .main > .block-container { padding: 2rem 3rem; }
    .stApp { background: linear-gradient(180deg, #0b2545 0%, #07203a 100%); color: white; }
    .stMarkdown, .stText, .stButton>button, .stTextInput>div>input {
        color: #ffffff !important;
    }
    .card {
      background: white;
      color: #0b2545;
      padding: 1rem;
      border-radius: 12px;
      box-shadow: rgba(2, 6, 23, 0.4) 0px 8px 24px;
    }
    .muted { color: #cbd5e1; }
    .source-badge { font-size: 12px; padding: 4px 8px; background:#eef2ff; color:#0b2545; border-radius:8px; display:inline-block;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utility functions
# -------------------------
def fetch_url_text(url: str, max_chars: int = 200_000) -> Tuple[str, str]:
    """
    Fetch HTML and extract visible text using BeautifulSoup.
    Returns (title, text).
    """
    headers = {"User-Agent": "RAG-Streamlit-App/1.0 (+https://example.com)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Attempt to extract a good textual body: article tags, main, then fallback to all p
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
    # trim extremely long pages
    texts = texts.strip()
    if len(texts) > max_chars:
        texts = texts[:max_chars]
    return title_tag, texts

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Returns list of chunks.
    """
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
# Initialize resources (lazy)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def init_chroma_client(persist_directory: str = CHROMA_DIR):
    # Use chroma local persistent directory
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    return client

@st.cache_resource(show_spinner=False)
def get_or_create_collection(client, name=CHROMA_COLLECTION_NAME):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)

# -------------------------
# Chroma helpers
# -------------------------
def upsert_page_to_chroma(collection, url: str, title: str, chunks: List[str], embeddings: List[List[float]]):
    """
    Upsert a page: ids, metadatas, documents, embeddings
    metas include url, title, chunk_index
    """
    ids = [f"{uuid.uuid4()}_{i}" for i in range(len(chunks))]
    metadatas = [{"url": url, "title": title, "chunk_index": i} for i in range(len(chunks))]
    documents = [textwrap.shorten(c, width=1000, placeholder="") for c in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

def retrieve_relevant_chunks(collection, query_embedding, top_k=TOP_K):
    """
    Query chroma collection and return hits as (document, metadata, distance).
    """
    results = collection.query(query_embeddings=[query_embedding], n=top_k, include=["documents", "metadatas", "distances"])
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    hits = [{"doc": d, "meta": m, "dist": dist} for d, m, dist in zip(docs, metas, dists)]
    return hits

# -------------------------
# Gemini call (HTTP)
# -------------------------
def call_gemini_generate(prompt: str, model: str = GEMINI_MODEL, temperature: float = 0.0, max_output_tokens: int = 800):
    """
    Call Gemini model via HTTP generate endpoint.
    NOTE: Use GEMINI_API_KEY env var for authentication.
    The function posts a JSON payload with prompt and returns the text output.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    endpoint = GEMINI_API_ENDPOINT.format(model=model)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # The request shape follows the Google GenAI HTTP pattern (basic).
    payload = {
        "prompt": {
            "text": prompt
        },
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    # Attempt to extract text result safely
    # NOTE: shape may vary across SDKs/versions. We try common fields.
    text = ""
    if "candidates" in j:
        text = j["candidates"][0].get("content", "")
    elif "output" in j:
        # Vertex AI style
        outs = j["output"]
        if isinstance(outs, list) and len(outs) > 0 and "content" in outs[0]:
            text = outs[0]["content"]
    else:
        # fallback: stringify
        text = j.get("text") or str(j)
    return text

# -------------------------
# Prompt engineering for RAG + citations
# -------------------------
def build_rag_prompt(question: str, retrieved: List[dict]) -> str:
    """
    Build a prompt for Gemini that includes retrieved context chunks and asks for a full explanation plus citations.
    We show the source metadata as [source_i] tags.
    """
    context_blocks = []
    for i, hit in enumerate(retrieved):
        meta = hit.get("meta", {})
        url = meta.get("url", "unknown")
        title = meta.get("title", "")
        chunk_id = meta.get("chunk_index", i)
        doc = hit.get("doc", "")
        # tag each chunk with a short source tag
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

# Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    url_input = st.text_input("Page URL", placeholder="https://example.com/article-or-doc", key="url_input")
    load_btn = st.button("Fetch & Index URL", key="index_btn")
    st.markdown("**Indexing status**")
    status_area = st.empty()

with col2:
    user_question = st.text_area("Your question about the page", height=140, placeholder="e.g., Summarize the main argument and list supporting evidence.")
    answer_btn = st.button("Ask AI", key="ask_btn")

# Initialize model & chroma
embed_model = load_embed_model()
client = init_chroma_client(CHROMA_DIR)
collection = get_or_create_collection(client, CHROMA_COLLECTION_NAME)

# Indexing flow
if load_btn:
    if not url_input:
        st.warning("Please enter a URL first.")
    else:
        status_area.info("Fetching URL...")
        try:
            title, page_text = fetch_url_text(url_input)
            status_area.success(f"Fetched: {title[:120]}")
            # chunk
            chunks = chunk_text(page_text)
            status_area.info(f"Text split into {len(chunks)} chunks. Generating embeddings...")
            # embed in batches
            embeddings = embed_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).tolist()
            status_area.info("Upserting into ChromaDB...")
            upsert_page_to_chroma(collection, url_input, title, chunks, embeddings)
            # persist chroma (DuckDB+parquet persist)
            client.persist()
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
            q_emb = embed_model.encode([user_question], convert_to_numpy=True)[0].tolist()
            hits = retrieve_relevant_chunks(collection, q_emb, top_k=TOP_K)
            if not hits or all(h["doc"] == "" for h in hits):
                st.info("No relevant content found for that URL in the vectorstore. Try indexing the page first.")
            else:
                # Build prompt for Gemini
                prompt = build_rag_prompt(user_question, hits)
                # Show retrieved snippets on UI
                st.markdown("### Retrieved context")
                for i, h in enumerate(hits):
                    meta = h["meta"]
                    st.write(f"**Source [{i+1}]** — {meta.get('title','')}")
                    st.write(f"<div class='card'>{h['doc'][:1000]}{'...' if len(h['doc'])>1000 else ''}</div>", unsafe_allow_html=True)
                st.markdown("---")
                st.info("Calling Gemini 2.5-Flash for answer generation...")
                try:
                    answer_text = call_gemini_generate(prompt=prompt, model=GEMINI_MODEL, temperature=0.0, max_output_tokens=800)
                except Exception as e:
                    st.error(f"Error calling Gemini: {e}")
                    answer_text = None

                if answer_text:
                    st.markdown("## AI Answer")
                    st.write(answer_text)
                    # try to extract sources mapping heuristically and display mapping:
                    st.markdown("### Source mapping (from retrieved hits)")
                    for i, h in enumerate(hits):
                        meta = h["meta"]
                        st.markdown(f"- [source:{i+1}] → {meta.get('title','')} — {meta.get('url')}")
