"""
RAG Backend — Proper Vector Database + Embeddings
Author: Suresh D R | AI Product Developer & Technology Mentor

This is REAL RAG:
- ChromaDB                → vector database (stores embeddings locally)
- OpenAI text-embedding-3-small → embedding model (1536 dimensions)
- OpenAI GPT-3.5          → LLM for answer generation

Why OpenAI embeddings?
- Production-grade quality — same model used by ChatGPT's retrieval
- 1536 dimensions — captures much richer semantic meaning
- text-embedding-3-small is cheap — $0.02 per million tokens
"""
# CI/CD test
import os
import uuid
import chromadb
from openai import OpenAI

# ── Embedding Config ───────────────────────────────────────────
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

def get_embedding_model():
    """Return the embedding model name — actual calls use OpenAI API"""
    return EMBEDDING_MODEL_NAME

def embed_texts(texts: list, openai_client: OpenAI) -> list:
    """
    Convert a list of text strings into embedding vectors
    using OpenAI text-embedding-3-small model.
    Returns a list of 1536-dim vectors.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=texts
    )
    return [item.embedding for item in response.data]

# ── ChromaDB Client (local persistent vector DB) ────────────────
def get_chroma_client(persist_dir="./chroma_db"):
    """Create or connect to local ChromaDB"""
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

# ── Text Chunking ────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    """
    Split text into overlapping chunks.

    Why overlap?
    If an answer spans two chunks — the overlap ensures the context
    is not lost at the boundary between chunks.

    chunk_size = 400 words per chunk
    overlap    = 80 words repeated from previous chunk
    """
    # Split by paragraphs first (double newline)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph exceeds chunk_size — save current and start new
        if len(current_words) + len(para_words) > chunk_size and current_words:
            chunk_text_str = " ".join(current_words)
            if chunk_text_str.strip():
                chunks.append(chunk_text_str.strip())
            # Keep last `overlap` words for context continuity
            current_words = current_words[-overlap:] + para_words
        else:
            current_words.extend(para_words)

    # Save the last chunk
    if current_words:
        chunk_text_str = " ".join(current_words)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str.strip())

    return chunks


# ── Index Document into ChromaDB ────────────────────────────────
def index_document(
    text: str,
    collection_name: str,
    source_name: str,
    chroma_client,
    openai_client: OpenAI
) -> dict:
    """
    Full indexing pipeline:
    1. Chunk the text
    2. Generate embeddings using OpenAI text-embedding-3-small
    3. Store chunks + embeddings + metadata in ChromaDB
    """
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No content to index", "chunks": 0}

    # Generate embeddings via OpenAI API
    embeddings = embed_texts(chunks, openai_client)

    # Get or recreate ChromaDB collection
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{
            "source": source_name,
            "chunk_index": i,
            "chunk_length": len(chunk)
        } for i, chunk in enumerate(chunks)]
    )

    return {
        "status": "indexed",
        "source": source_name,
        "total_chunks": len(chunks),
        "collection": collection_name,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIMENSIONS
    }


# ── Retrieve Relevant Chunks ─────────────────────────────────────
def retrieve_chunks(
    query: str,
    collection_name: str,
    chroma_client,
    openai_client: OpenAI,
    top_k: int = 5
) -> list:
    """
    Vector similarity search using OpenAI embeddings:
    1. Embed the query using text-embedding-3-small
    2. ChromaDB finds the top_k most similar chunks (cosine similarity)
    3. Returns chunks with their similarity scores
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return []

    query_embedding = embed_texts([query], openai_client)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if results["documents"] and results["documents"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 - dist
            retrieved.append({
                "chunk_index": i + 1,
                "content": doc,
                "source": meta.get("source", "unknown"),
                "similarity_score": round(similarity, 4),
                "chunk_number": meta.get("chunk_index", i)
            })
    return retrieved


# ── Generate Answer with RAG ─────────────────────────────────────
def generate_rag_answer(
    question: str,
    retrieved_chunks: list,
    openai_client: OpenAI
) -> str:
    """
    Final step of RAG:
    - Build prompt with retrieved chunks as context
    - Send to GPT-3.5 to generate grounded answer
    - LLM ONLY uses the retrieved context — not general knowledge
    """
    if not retrieved_chunks:
        return "I could not find relevant information in the document to answer this question."

    # Build context from retrieved chunks
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source chunk {chunk['chunk_index']} "
            f"| Relevance: {chunk['similarity_score']:.2f}]\n"
            f"{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # RAG System Prompt
    system_prompt = """You are a precise sales data analyst assistant.

Answer the user's question using ONLY the information provided in the context below.

Rules:
- Be specific with numbers, names, percentages, and dates from the context
- If the answer involves calculations (totals, averages), do them accurately
- If the context does not contain enough information, say so clearly
- Do not add information not present in the context
- Format numbers clearly with Rs prefix and lakh/crore units as in the source
- Structure your answer clearly with bullet points when listing multiple items

CONTEXT FROM SALES REPORT:
{context}""".format(context=context)

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=600,
        temperature=0.1  # very low temperature = more factual, less creative
    )

    return response.choices[0].message.content


# ── Full RAG Pipeline ────────────────────────────────────────────
def run_rag_pipeline(
    question: str,
    collection_name: str,
    chroma_client,
    openai_client: OpenAI,
    top_k: int = 5
) -> dict:
    """Complete RAG pipeline: retrieve → generate → return"""
    retrieved = retrieve_chunks(
        query=question,
        collection_name=collection_name,
        chroma_client=chroma_client,
        openai_client=openai_client,
        top_k=top_k
    )
    answer = generate_rag_answer(
        question=question,
        retrieved_chunks=retrieved,
        openai_client=openai_client
    )
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "num_chunks_used": len(retrieved),
        "top_similarity": retrieved[0]["similarity_score"] if retrieved else 0
    }
# retry
