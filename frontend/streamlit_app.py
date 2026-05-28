"""
Sales RAG Assistant — Streamlit Frontend
Author: Suresh D R | AI Product Developer & Technology Mentor

Proper RAG with:
- ChromaDB vector database
- sentence-transformers embedding model (all-MiniLM-L6-v2)
- OpenAI GPT-3.5 for answer generation
- Upload your own TXT file or use sample sales data
"""

import streamlit as st
import os
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from rag_engine import (
    get_embedding_model,
    get_chroma_client,
    index_document,
    run_rag_pipeline,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS
)
from openai import OpenAI

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .chunk-box {
        background: #1a2332;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.6;
    }
    .score-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .how-it-works {
        background: #0f2027;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid #1e3a5f;
    }
    .step-num {
        color: #3b82f6;
        font-weight: 800;
        font-size: 16px;
    }
    .info-box {
        background: #162032;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #2d3f5a;
        margin: 6px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models Once (cached) ──────────────────────────────────
@st.cache_resource(show_spinner="Setting up vector database...")
def load_models():
    chroma_client = get_chroma_client(persist_dir="/tmp/chroma_db")
    return chroma_client

# ── Default Sales Data ─────────────────────────────────────────
def load_default_data():
    default_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'sales_report_q1_2024.txt'
    )
    if os.path.exists(default_path):
        with open(default_path, 'r') as f:
            return f.read()
    return None

# ── Session State ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "index_info" not in st.session_state:
    st.session_state.index_info = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None

# Load models
chroma_client = load_models()

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Sales RAG Assistant")
    st.markdown("*Proper RAG — Vector DB + Embeddings*")
    st.divider()

    # API Key
    st.markdown("### 🔑 OpenAI API Key")
    api_key = st.text_input(
        "API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        label_visibility="collapsed"
    )
    if api_key:
        st.session_state.api_key = api_key
        st.success("✅ API key set")

    st.divider()

    # Data Upload
    st.markdown("### 📁 Upload Your Document")
    st.caption("Upload a TXT file — sales reports, HR docs, policies, etc.")

    uploaded_file = st.file_uploader(
        "Upload TXT file",
        type=["txt"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.doc_text = text
        collection_name = uploaded_file.name.replace(".", "_").replace(" ", "_").lower()
        st.session_state.collection_name = collection_name
        st.session_state.indexed = False
        st.session_state.chat_history = []
        st.success(f"✅ File loaded: {len(text.split())} words")

    # Load default
    st.markdown("**Or use sample data:**")
    if st.button("📊 Load Q1 Sales Report", use_container_width=True):
        text = load_default_data()
        if text:
            st.session_state.doc_text = text
            st.session_state.collection_name = "sales_q1_2024"
            st.session_state.indexed = False
            st.session_state.chat_history = []
            st.success(f"✅ Sales report loaded: {len(text.split())} words")
        else:
            st.error("sales_report_q1_2024.txt not found in data/ folder")

    # Index button
    if st.session_state.doc_text and not st.session_state.indexed:
        st.divider()
        if st.button("🔍 Index Document into Vector DB", use_container_width=True, type="primary"):
            if not st.session_state.api_key:
                st.error("Enter your OpenAI API key first!")
            else:
                with st.spinner("Chunking → Embedding via OpenAI → Storing in ChromaDB..."):
                    oai_client = OpenAI(api_key=st.session_state.api_key)
                    info = index_document(
                        text=st.session_state.doc_text,
                        collection_name=st.session_state.collection_name,
                        source_name=st.session_state.collection_name,
                        chroma_client=chroma_client,
                        openai_client=oai_client
                    )
                    st.session_state.index_info = info
                    st.session_state.indexed = True
                st.success(f"✅ Indexed {info['total_chunks']} chunks into ChromaDB!")
                st.rerun()

    # Show index info
    if st.session_state.indexed and st.session_state.index_info:
        info = st.session_state.index_info
        st.divider()
        st.markdown("### 📦 Vector DB Status")
        st.markdown(f"""
        <div class="info-box">
        ✅ <b>Status:</b> Indexed<br>
        📄 <b>Chunks:</b> {info['total_chunks']}<br>
        🧠 <b>Embedding:</b> text-embedding-3-small<br>
        📐 <b>Dimensions:</b> 1536<br>
        🗄️ <b>DB:</b> ChromaDB (local)
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Top K slider
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=5)
    st.caption("More chunks = more context but slower")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Suresh D R | AI Product Developer")

# ── MAIN ───────────────────────────────────────────────────────
st.markdown("# 📊 Sales RAG Assistant")
st.markdown("**Proper RAG** — ChromaDB vector database + sentence-transformer embeddings + GPT-3.5")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Document Preview", "🔬 How RAG Works Here"])

# ── TAB 1: CHAT ────────────────────────────────────────────────
with tab1:
    if not st.session_state.doc_text:
        st.info("👈 Upload a TXT file or load the sample Q1 Sales Report from the sidebar.")

        st.markdown("### 💡 Sample questions you can ask after loading the sales report:")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            - What was the total Q1 2024 revenue?
            - How did February perform compared to January?
            - Which sales rep had the highest revenue in Q1?
            - What was the SaaS revenue in March?
            """)
        with c2:
            st.markdown("""
            - Which region had the highest growth?
            - How many deals were lost to Salesforce?
            - What is the pipeline at end of Q1?
            - What are the Q2 targets?
            """)

    elif not st.session_state.indexed:
        st.warning("👈 Click **Index Document into Vector DB** in the sidebar to start chatting.")
        st.markdown("This step:")
        st.markdown("""
        1. Splits your document into chunks
        2. Converts each chunk into a vector (384 numbers) using the embedding model
        3. Stores all vectors in ChromaDB
        4. You can then ask questions — ChromaDB finds the most relevant chunks instantly
        """)

    else:
        if not st.session_state.api_key:
            st.warning("👈 Enter your OpenAI API key in the sidebar to start chatting.")
        else:
            # Show chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "chunks" in msg:
                        with st.expander(
                            f"🔍 {len(msg['chunks'])} chunks retrieved from ChromaDB "
                            f"(top similarity: {msg['top_similarity']:.2f})"
                        ):
                            for chunk in msg["chunks"]:
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<span class="score-badge">Chunk {chunk["chunk_number"]+1} | '
                                    f'Similarity: {chunk["similarity_score"]:.4f}</span><br>'
                                    f'{chunk["content"][:400]}{"..." if len(chunk["content"]) > 400 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

            # Chat input
            question = st.chat_input("Ask a question about the sales report...")

            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching vector DB → retrieving chunks → generating answer..."):
                        t_start = time.time()
                        client = OpenAI(api_key=st.session_state.api_key)
                        result = run_rag_pipeline(
                            question=question,
                            collection_name=st.session_state.collection_name,
                            chroma_client=chroma_client,
                            openai_client=client,
                            top_k=top_k
                        )
                        t_end = time.time()

                    st.markdown(result["answer"])

                    # Show metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Chunks Retrieved", result["num_chunks_used"])
                    m2.metric("Top Similarity", f"{result['top_similarity']:.3f}")
                    m3.metric("Response Time", f"{t_end - t_start:.1f}s")

                    if result["retrieved_chunks"]:
                        with st.expander(
                            f"🔍 {result['num_chunks_used']} chunks retrieved "
                            f"(top similarity: {result['top_similarity']:.4f})"
                        ):
                            for chunk in result["retrieved_chunks"]:
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<span class="score-badge">Chunk {chunk["chunk_number"]+1} | '
                                    f'Similarity: {chunk["similarity_score"]:.4f}</span><br>'
                                    f'{chunk["content"][:400]}{"..." if len(chunk["content"]) > 400 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "chunks": result["retrieved_chunks"],
                    "top_similarity": result["top_similarity"]
                })

# ── TAB 2: DOCUMENT PREVIEW ────────────────────────────────────
with tab2:
    if st.session_state.doc_text:
        text = st.session_state.doc_text
        words = text.split()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Words", f"{len(words):,}")
        c2.metric("Total Lines", f"{len(text.splitlines()):,}")
        c3.metric("Chunks After Indexing",
                  st.session_state.index_info["total_chunks"]
                  if st.session_state.index_info else "Not indexed yet")

        st.markdown("### 📄 Document Content")
        st.text_area("", text, height=500, label_visibility="collapsed")
    else:
        st.info("Load a document from the sidebar to preview it.")

# ── TAB 3: HOW RAG WORKS ───────────────────────────────────────
with tab3:
    st.markdown("### 🔬 How This RAG System Works — Step by Step")

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Step 1 — You upload a document</span><br>
    A TXT file is loaded into memory. This is your knowledge source.
    The AI will ONLY answer from this document — nothing else.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Step 2 — Document is chunked</span><br>
    The document is split into overlapping chunks of ~400 words each.
    Overlap of 80 words ensures no answer is lost at a chunk boundary.<br><br>
    Example: A 3,000 word sales report → approximately 12 chunks.<br>
    Each chunk covers one topic or section.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Step 3 — Chunks are embedded into vectors</span><br>
    OpenAI's <b>text-embedding-3-small</b> model converts each chunk into
    a vector of 1536 numbers. This is called an embedding.<br><br>
    What is an embedding vector?<br>
    "Revenue in March was Rs 1.79 crore" → [0.23, -0.41, 0.88, ... 1536 numbers]<br><br>
    Similar meaning → vectors point in similar directions in 1536-dimensional space.<br>
    "March sales" and "third month performance" will have very similar vectors.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Step 4 — Vectors are stored in ChromaDB</span><br>
    ChromaDB is a local vector database. It stores:<br>
    - The original text chunks<br>
    - Their 384-dim embedding vectors<br>
    - Metadata (source, chunk number)<br><br>
    ChromaDB uses HNSW index for fast approximate nearest neighbour search.
    It can search through thousands of chunks in milliseconds.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Step 5 — You ask a question</span><br>
    Your question is also embedded using the same model:<br>
    "What was March revenue?" → [0.19, -0.38, 0.91, ... 384 numbers]
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Step 6 — ChromaDB does vector similarity search</span><br>
    ChromaDB compares your question vector against all stored chunk vectors.<br>
    Similarity is measured using <b>cosine similarity</b> — angle between vectors.<br><br>
    Score = 1.0 → identical meaning<br>
    Score = 0.8+ → very similar meaning<br>
    Score = 0.5  → somewhat related<br>
    Score below 0.3 → not related<br><br>
    Top 5 most similar chunks are returned — these are the most relevant parts of the document.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Step 7 — GPT-3.5 generates the answer</span><br>
    The 5 retrieved chunks + your question go to GPT-3.5.<br>
    System prompt instructs the model to answer ONLY from the retrieved context.<br>
    GPT-3.5 reads the chunks and writes a precise, grounded answer.<br><br>
    This prevents hallucination — the model cannot make up numbers
    because it only has the retrieved chunks in front of it.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
    <span class="step-num">Why this is better than keyword search</span><br>
    Keyword search: "March revenue" only finds chunks containing those exact words.<br>
    Vector search: "March revenue" also finds "third month performance", "Q1 final month",
    "month-on-month March increase" — because they mean the same thing.<br><br>
    This is called <b>semantic search</b> — finding similar meaning, not just matching words.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Tech Stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Embedding Model**\ntext-embedding-3-small\n1536 dimensions\nOpenAI API\nProduction-grade quality")
    with c2:
        st.info("**Vector Database**\nChromaDB\nLocal persistent\nCosine similarity\nHNSW index")
    with c3:
        st.info("**LLM**\nGPT-3.5-turbo\nOpenAI API\nTemp = 0.1\nGrounded answers")
