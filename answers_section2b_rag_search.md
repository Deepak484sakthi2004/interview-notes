# Section 2B: RAG, OpenSearch, LangChain & Prompt Engineering — Interview Answers

**Candidate: Deepaksakthi**

---

## Q46. Explain your RAG pipeline architecture end-to-end — from document ingestion to response generation.

**Answer:**

My RAG pipeline for the Knowledge Chatbot followed a multi-stage architecture that I designed to serve over 500 documents with sub-200ms retrieval latency.

**Stage 1 — Document Ingestion & Preprocessing:**
Documents arrive in various formats (PDF, DOCX, HTML, Markdown). I built a preprocessing layer using LangChain's document loaders that normalizes everything into plain text. Metadata extraction happens here — source, timestamp, category, and document ID are tagged to every chunk.

**Stage 2 — Chunking:**
I used recursive character text splitting with chunk sizes of 512 tokens and 50-token overlap. This preserves sentence boundaries and ensures context continuity across chunks. Each chunk retains its parent document metadata.

**Stage 3 — Embedding & Indexing:**
Chunks are embedded using a sentence-transformer model (all-MiniLM-L6-v2 for speed, or all-mpnet-base-v2 for quality). The dense vectors go into ChromaDB. Simultaneously, I maintain a keyword index (BM25) over the same chunks for hybrid retrieval.

**Stage 4 — Query Processing:**
When a user query arrives, it passes through query preprocessing — spelling correction, expansion, and embedding. The query is dispatched to both the dense retriever (ChromaDB similarity search) and the sparse retriever (BM25 keyword search) in parallel.

**Stage 5 — Hybrid Retrieval & Re-ranking:**
Results from both retrievers are fused using Reciprocal Rank Fusion (RRF). The top candidates then pass through a cross-encoder re-ranker that scores query-document pairs more precisely. This gives us the final top-k (typically k=5) most relevant chunks.

**Stage 6 — Prompt Construction & Generation:**
The retrieved chunks are injected into a structured prompt template:

```python
prompt_template = """You are a helpful knowledge assistant.
Use ONLY the following context to answer the question.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
```

The prompt goes to the LLM (GPT-3.5-turbo or GPT-4 depending on complexity), and the response is streamed back through the Flask API.

**Stage 7 — Observability:**
Every request is traced through LangFuse — embedding latency, retrieval scores, token counts, and LLM response quality are all logged for monitoring and continuous improvement.

The architecture in summary:

```
User Query → Query Preprocessing → [Dense Retriever (ChromaDB) || Sparse Retriever (BM25)]
  → Reciprocal Rank Fusion → Cross-Encoder Re-ranking → Top-K Chunks
  → Prompt Construction → LLM Generation → Response + LangFuse Trace
```

This modular design allowed me to independently tune each stage. For example, I could swap the embedding model or adjust the re-ranker threshold without touching the rest of the pipeline.

---

## Q47. What is hybrid retrieval (dense + keyword)? How did you combine the scores from both?

**Answer:**

Hybrid retrieval combines two fundamentally different search paradigms to overcome the weaknesses of each individual approach.

**Dense Retrieval** uses embedding models to encode both queries and documents into high-dimensional vector spaces. Similarity is computed via cosine similarity or dot product. This excels at semantic understanding — it can match "automobile" to "car" even though the words are different. However, it struggles with exact-match terms, rare keywords, product codes, or domain-specific acronyms.

**Keyword (Sparse) Retrieval** uses algorithms like BM25 that rely on term frequency and inverse document frequency. It excels at exact-match scenarios — searching for "ERR_CONN_REFUSED" will find that exact string. However, it misses semantic relationships entirely.

**Why Hybrid?** In my knowledge chatbot, users asked questions ranging from conceptual ("how does authentication work?") to exact-match ("what is error code 4021?"). Neither retriever alone could handle both well.

**Score Combination — Reciprocal Rank Fusion (RRF):**

I used RRF because it elegantly combines ranked lists without needing to normalize raw scores (which are on completely different scales between BM25 and cosine similarity):

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Combine multiple ranked lists using RRF.
    k=60 is the standard smoothing constant.
    """
    fused_scores = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.metadata["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by fused score descending
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# Usage
dense_results = chroma_collection.query(query_embedding, n_results=20)
sparse_results = bm25_retriever.get_relevant_documents(query, top_k=20)

fused = reciprocal_rank_fusion([dense_results, sparse_results])
top_candidates = fused[:10]  # Pass to re-ranker
```

**Why RRF over weighted sum?** A weighted sum (e.g., `0.7 * dense_score + 0.3 * bm25_score`) requires normalizing scores to the same scale, which is fragile. BM25 scores can range from 0 to 30+, while cosine similarity is 0-1. RRF avoids this problem entirely by only considering rank positions.

**I also experimented with a weighted approach:**

```python
# Normalized weighted combination (alternative)
dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
combined = alpha * dense_norm + (1 - alpha) * sparse_norm  # alpha=0.7 worked best
```

**Results:** Hybrid retrieval improved recall@10 by approximately 15-20% compared to dense-only retrieval in my evaluations, particularly on queries containing technical terms and error codes.

---

## Q48. What re-ranking strategy did you use? BM25 + cross-encoder? How did it improve results?

**Answer:**

I implemented a two-stage retrieval pipeline: fast initial retrieval followed by precise re-ranking.

**Stage 1 — Fast Retrieval (BM25 + Dense):**
The hybrid retriever fetches the top 20-30 candidates quickly. These retrievers are optimized for recall (finding all potentially relevant documents) rather than precision.

**Stage 2 — Cross-Encoder Re-ranking:**
The top candidates are passed through a cross-encoder model that jointly encodes the query and each document together. Unlike bi-encoders (used in stage 1) that encode query and document independently, cross-encoders can capture fine-grained interactions between query tokens and document tokens.

```python
from sentence_transformers import CrossEncoder

# Load cross-encoder model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

def rerank_documents(query, documents, top_k=5):
    """Re-rank documents using cross-encoder."""
    # Create query-document pairs
    pairs = [(query, doc.page_content) for doc in documents]
    
    # Score all pairs
    scores = reranker.predict(pairs)
    
    # Sort by score and return top-k
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs[:top_k]

# Pipeline
initial_candidates = hybrid_retrieve(query, top_k=25)
reranked = rerank_documents(query, initial_candidates, top_k=5)
```

**Why this specific cross-encoder?** I used `ms-marco-MiniLM-L-6-v2` because it is trained on the MS MARCO passage ranking dataset, which aligns well with question-answering style retrieval. It provides a strong balance between accuracy and speed — re-ranking 25 documents takes roughly 30-50ms on a GPU and 80-120ms on CPU.

**How it improved results:**

| Metric | Without Re-ranking | With Re-ranking | Improvement |
|--------|-------------------|-----------------|-------------|
| MRR@5 | 0.62 | 0.78 | +25.8% |
| NDCG@5 | 0.58 | 0.74 | +27.6% |
| Precision@3 | 0.65 | 0.82 | +26.2% |

**Key insight:** The cross-encoder rescued many cases where the correct document was ranked 8th-15th by the initial retriever but got promoted to top-3 after re-ranking. This was especially impactful for ambiguous queries where keyword overlap was misleading.

**Trade-offs:**
- Cross-encoders are O(n) per query (must score each candidate), so you cannot run them over the entire corpus — hence the two-stage approach.
- I capped re-ranking at 25 candidates to keep total latency under 200ms.
- For production, I batched the cross-encoder predictions and used ONNX runtime for faster inference.

**Alternative considered:** I also evaluated Cohere Rerank API as a managed alternative. It performed comparably but added network latency and cost, so I kept the self-hosted cross-encoder.

---

## Q49. How did you chunk the 500+ documents? What chunking strategy (fixed size, semantic, recursive) and why?

**Answer:**

Chunking strategy is one of the most impactful decisions in a RAG pipeline. I used **recursive character text splitting** as my primary strategy, after experimenting with several approaches.

**Strategies I evaluated:**

**1. Fixed-Size Chunking:**
```python
# Simple but crude
chunks = [text[i:i+512] for i in range(0, len(text), 512)]
```
Problem: Cuts mid-sentence and mid-paragraph, breaking semantic coherence. A question about "authentication flow" might get split across two chunks, and neither chunk alone has the complete answer.

**2. Semantic Chunking:**
Splits at semantic boundaries by measuring embedding similarity between consecutive sentences. When similarity drops below a threshold, it creates a new chunk.
Problem: Computationally expensive at ingestion time, chunk sizes are highly variable (some too small, some too large), and it was overkill for my document types.

**3. Recursive Character Text Splitting (My choice):**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

chunks = splitter.split_documents(documents)
```

**Why recursive?** It tries to split on the most meaningful boundary first (double newline = paragraph break), then falls back to less meaningful boundaries (newline, sentence, word, character). This preserves semantic coherence while maintaining consistent chunk sizes.

**My specific configuration and rationale:**

- **chunk_size=512 tokens:** After testing 256, 512, and 1024, I found 512 was the sweet spot. Smaller chunks had too little context (answer fragments), larger chunks diluted relevance and consumed too much of the LLM context window.

- **chunk_overlap=50 tokens:** Overlap ensures that if an answer spans a chunk boundary, at least part of it appears in both chunks. 50 tokens (~10% overlap) was sufficient without creating excessive redundancy.

- **Separator hierarchy:** `["\n\n", "\n", ". ", " ", ""]` — this respects document structure. Technical documents in my corpus were well-structured with paragraph breaks, so most splits happened at `\n\n`.

**Additional strategies I layered on:**

**Parent-child chunking for long documents:**
```python
# Store small chunks for retrieval, but return parent chunk for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

parent_chunks = parent_splitter.split_documents(docs)
for parent in parent_chunks:
    children = child_splitter.split_documents([parent])
    for child in children:
        child.metadata["parent_id"] = parent.metadata["id"]
```

This way, retrieval matches on the precise child chunk, but the LLM receives the broader parent chunk for more context.

**Metadata enrichment per chunk:**
Every chunk carries: `source_file`, `chunk_index`, `total_chunks`, `section_heading`, and `document_category`. This enables filtered retrieval (e.g., only search within "API documentation" category).

**Results:** Recursive splitting with parent-child retrieval improved answer quality by ~18% (measured by human evaluation) compared to naive fixed-size chunking.

---

## Q50. How did you handle documents that update frequently? Did you have an incremental indexing pipeline?

**Answer:**

Yes, I built an incremental indexing pipeline because re-indexing 500+ documents on every change was wasteful and caused downtime.

**The Problem:**
Some documents in the knowledge base were updated daily (release notes, FAQ pages, internal wikis). Naive full re-indexing meant: (1) downtime during re-indexing, (2) wasted compute re-embedding unchanged documents, and (3) stale data if indexing was infrequent.

**My Incremental Indexing Architecture:**

```python
import hashlib
from datetime import datetime

class IncrementalIndexer:
    def __init__(self, chroma_client, embedding_model):
        self.collection = chroma_client.get_or_create_collection("knowledge_base")
        self.embedder = embedding_model
        self.doc_registry = {}  # doc_id -> {hash, chunk_ids, last_updated}
    
    def compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
    
    def index_document(self, doc_id: str, content: str, metadata: dict):
        new_hash = self.compute_hash(content)
        
        # Check if document has changed
        if doc_id in self.doc_registry:
            if self.doc_registry[doc_id]["hash"] == new_hash:
                return "skipped"  # No changes
            else:
                # Delete old chunks before re-indexing
                old_chunk_ids = self.doc_registry[doc_id]["chunk_ids"]
                self.collection.delete(ids=old_chunk_ids)
        
        # Chunk and embed the new/updated document
        chunks = self.splitter.split_text(content)
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.encode(chunks)
        
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{**metadata, "doc_id": doc_id, "chunk_idx": i} 
                       for i in range(len(chunks))]
        )
        
        # Update registry
        self.doc_registry[doc_id] = {
            "hash": new_hash,
            "chunk_ids": chunk_ids,
            "last_updated": datetime.utcnow().isoformat()
        }
        return "indexed"
```

**Change Detection Strategies:**

1. **Hash-based:** Compare SHA-256 hash of document content. If unchanged, skip. This was my primary method.
2. **Timestamp-based:** For documents from APIs or databases, I compared `last_modified` timestamps.
3. **Webhook-driven:** For wiki-style sources, I set up webhooks that trigger re-indexing of specific documents on update.

**Pipeline Scheduling:**

```
Cron Job (every 15 min) → Scan Sources → Detect Changes (hash comparison)
  → Changed docs only → Re-chunk → Re-embed → Update ChromaDB
  → Update BM25 index → Log to LangFuse
```

**Handling deletions:** When a document is removed from the source, the indexer deletes all associated chunks from both ChromaDB and the BM25 index using the stored `chunk_ids`.

**Zero-downtime strategy:** ChromaDB supports atomic operations, so I could delete old chunks and insert new ones without taking the service offline. For larger migrations, I used a blue-green approach — build a new collection, then swap the alias.

**BM25 index updates:** Since BM25 (using `rank_bm25` library) is an in-memory structure, I serialized it with pickle after each update and loaded it on service restart. For the incremental update, I rebuilt the BM25 index from the current ChromaDB documents every cycle (it was fast enough for 500 docs).

---

## Q51. What embedding model did you use? Why that specific one?

**Answer:**

I primarily used **`all-MiniLM-L6-v2`** from the sentence-transformers library, with **`all-mpnet-base-v2`** as a higher-quality alternative for accuracy-critical use cases.

**Why `all-MiniLM-L6-v2`:**

| Property | Value |
|----------|-------|
| Embedding Dimension | 384 |
| Model Size | 80MB |
| Speed | ~14,000 sentences/sec on GPU |
| Max Sequence Length | 256 tokens |
| MTEB Benchmark Score | ~0.63 |

1. **Latency requirements:** With sub-200ms end-to-end target, embedding the query had to be fast. MiniLM-L6-v2 encodes a query in ~5ms on GPU, leaving budget for retrieval and re-ranking.

2. **Storage efficiency:** 384-dimensional vectors are half the size of 768-dim alternatives. With 500+ documents chunked into ~5,000 chunks, this kept the ChromaDB index compact and fast.

3. **Quality-to-speed ratio:** On the MTEB benchmark, it scores within 5-8% of much larger models like `all-mpnet-base-v2` while being 5x faster. For retrieval (where the cross-encoder re-ranker corrects ranking errors), this trade-off was favorable.

```python
from sentence_transformers import SentenceTransformer

# Primary model for production
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embedding at ingestion time
doc_embeddings = model.encode(
    chunks,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True  # For cosine similarity
)

# Query-time embedding
query_embedding = model.encode(query, normalize_embeddings=True)
```

**When I used `all-mpnet-base-v2` instead:**
For a subset of the knowledge base containing complex technical documentation where retrieval precision was critical and latency budget was more relaxed (API queries with 500ms SLA), I used mpnet. It produces 768-dimensional embeddings and scores higher on semantic similarity tasks.

**Models I evaluated and rejected:**

- **OpenAI `text-embedding-ada-002`:** Excellent quality (1536 dims), but added API latency (~100-200ms per call) and cost. For a self-hosted chatbot handling thousands of queries daily, this was not viable.
- **`e5-large-v2`:** Strong performance but 1024 dimensions and slower inference. The marginal quality gain did not justify the compute cost.
- **`BGE-small-en`:** Comparable to MiniLM but I found MiniLM performed slightly better on my specific domain after testing with a held-out evaluation set.

**Domain adaptation consideration:**
I considered fine-tuning the embedding model on domain-specific data using contrastive learning (positive pairs from actual user queries and their correct document matches). I created a small fine-tuning dataset of ~200 query-document pairs, but the improvement was marginal (~2% on recall@10) because the domain vocabulary was already well-represented in the pretrained model. If working with highly specialized domains (legal, medical), fine-tuning would be essential.

---

## Q52. How did you evaluate retrieval quality? Did you use metrics like MRR, NDCG, or recall@k?

**Answer:**

Yes, I built a systematic evaluation pipeline using multiple retrieval metrics. This was critical for making data-driven decisions about chunking, embedding models, and retrieval strategies.

**Evaluation Dataset Creation:**
I curated a golden test set of 150 query-document pairs:
- 50 from actual user queries (logged from the chatbot)
- 50 synthetically generated (using an LLM to create questions from documents)
- 50 edge cases (multi-hop questions, negation, exact-match terms)

Each query was annotated with relevant document chunk IDs and relevance grades (0=irrelevant, 1=partially relevant, 2=highly relevant).

**Metrics I Used:**

```python
import numpy as np

def recall_at_k(retrieved_ids, relevant_ids, k):
    """What fraction of relevant docs appear in top-k?"""
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / len(relevant) if relevant else 0

def mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank — how early does the first relevant doc appear?"""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevance_grades, k):
    """Normalized Discounted Cumulative Gain — accounts for graded relevance."""
    dcg = sum(
        relevance_grades.get(doc_id, 0) / np.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], 1)
    )
    # Ideal DCG
    ideal_scores = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = sum(score / np.log2(rank + 1) for rank, score in enumerate(ideal_scores, 1))
    return dcg / idcg if idcg > 0 else 0

# Evaluation loop
results = {"recall@5": [], "recall@10": [], "mrr": [], "ndcg@5": []}
for query, relevant_ids, relevance_grades in eval_dataset:
    retrieved = retriever.retrieve(query, top_k=10)
    retrieved_ids = [doc.metadata["id"] for doc in retrieved]
    
    results["recall@5"].append(recall_at_k(retrieved_ids, relevant_ids, 5))
    results["recall@10"].append(recall_at_k(retrieved_ids, relevant_ids, 10))
    results["mrr"].append(mrr(retrieved_ids, relevant_ids))
    results["ndcg@5"].append(ndcg_at_k(retrieved_ids, relevance_grades, 5))

for metric, values in results.items():
    print(f"{metric}: {np.mean(values):.3f}")
```

**Results across configurations:**

| Configuration | Recall@5 | Recall@10 | MRR | NDCG@5 |
|--------------|----------|-----------|-----|--------|
| Dense only | 0.68 | 0.79 | 0.62 | 0.58 |
| BM25 only | 0.55 | 0.71 | 0.51 | 0.48 |
| Hybrid (RRF) | 0.78 | 0.88 | 0.71 | 0.67 |
| Hybrid + Re-rank | 0.85 | 0.88 | 0.78 | 0.74 |

**End-to-End Evaluation (Answer Quality):**
Beyond retrieval metrics, I also evaluated the final LLM-generated answers using:
- **Faithfulness:** Does the answer stay true to the retrieved context? (Measured via LLM-as-judge)
- **Answer relevance:** Does the answer actually address the question?
- **Context relevance:** Are the retrieved chunks relevant to the query?

I used the RAGAS framework for some of these automated evaluations, supplemented by weekly manual reviews of 20 random production queries.

---

## Q53. What was your strategy for handling queries that have no relevant documents in the knowledge base?

**Answer:**

This is a critical problem in RAG systems — when the knowledge base does not contain an answer, the system should gracefully acknowledge it rather than hallucinate. I implemented a multi-layered approach.

**Layer 1 — Retrieval Confidence Thresholding:**

```python
def retrieve_with_confidence(query, threshold=0.35):
    results = hybrid_retrieve(query, top_k=5)
    
    # Check if top result meets minimum relevance threshold
    top_score = results[0].score if results else 0
    
    if top_score < threshold:
        return {
            "status": "no_relevant_docs",
            "message": "I don't have information about this in my knowledge base.",
            "suggestion": classify_query_intent(query)
        }
    
    # Also check score gap — if all results have similar low scores,
    # likely no single doc is truly relevant
    scores = [r.score for r in results[:5]]
    if max(scores) - min(scores) < 0.05 and max(scores) < 0.5:
        return {"status": "low_confidence", "docs": results}
    
    return {"status": "confident", "docs": results}
```

**Layer 2 — Prompt-Level Guardrails:**
The system prompt explicitly instructs the LLM to refuse when context is insufficient:

```python
system_prompt = """You are a knowledge assistant. Answer based ONLY on the provided context.

CRITICAL RULES:
1. If the context does not contain information to answer the question, respond with:
   "I don't have enough information in my knowledge base to answer this question."
2. NEVER make up information not present in the context.
3. If you can partially answer, do so and clearly state what information is missing.
4. Suggest related topics the user might want to explore if available."""
```

**Layer 3 — Query Classification:**
Before retrieval, I classify the query to detect out-of-scope requests:

```python
def classify_query_intent(query):
    """Classify if query is in-scope for the knowledge base."""
    # Simple keyword-based classifier + LLM fallback
    out_of_scope_patterns = [
        "weather", "stock price", "personal opinion",
        "write me a poem", "tell me a joke"
    ]
    
    for pattern in out_of_scope_patterns:
        if pattern in query.lower():
            return "out_of_scope"
    
    # For ambiguous cases, use lightweight LLM classification
    return llm_classify(query, categories=["in_scope", "out_of_scope", "ambiguous"])
```

**Layer 4 — Fallback Strategies:**
When no relevant documents are found, instead of a dead end, I provide:
1. **Related topics:** "I couldn't find information about X, but I have documents about Y and Z."
2. **Escalation path:** "Would you like me to flag this to the support team?"
3. **Query reformulation suggestion:** "Try rephrasing your question. For example, instead of 'how to fix the issue', try 'troubleshooting steps for error code XYZ'."

**Layer 5 — Monitoring and Feedback Loop:**
Every "no relevant docs" response is logged in LangFuse. I reviewed these weekly to identify knowledge gaps and added new documents to fill them. This turned failed queries into a continuous improvement signal.

Over three months, this feedback loop reduced the "no answer" rate from approximately 22% to 8% of all queries.

---

## Q54. How did you achieve sub-200ms retrieval latency? What optimizations did you apply?

**Answer:**

Achieving sub-200ms end-to-end latency required optimizations at every stage of the pipeline. Here is the breakdown of my latency budget and the optimizations I applied.

**Latency Budget (target: <200ms total):**
```
Query embedding:     ~5ms
Dense retrieval:     ~10ms
BM25 retrieval:      ~5ms
Score fusion (RRF):  ~2ms
Re-ranking:          ~50-80ms
Prompt construction: ~3ms
Buffer:              ~95ms
Total:               <200ms (retrieval only, excluding LLM generation)
```

Note: The sub-200ms target was for retrieval. LLM generation adds 500-2000ms but is streamed to the user.

**Optimization 1 — Embedding Model Selection:**
I chose `all-MiniLM-L6-v2` specifically for its speed. Query embedding in ~5ms versus 20-50ms for larger models.

```python
# Pre-load model at startup (not per request)
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Normalize at ingestion time so retrieval only needs dot product
embeddings = model.encode(chunks, normalize_embeddings=True)
```

**Optimization 2 — ChromaDB Configuration:**
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 16,           # Connections per node (default 16)
        "hnsw:ef_construction": 200,  # Build-time quality
        "hnsw:ef": 50,          # Query-time quality-speed trade-off
    }
)
```

Key tuning: `hnsw:ef=50` (search-time parameter) — lower values mean faster but less accurate search. I tested values from 20 to 200 and found 50 gave >99% recall compared to exact search while keeping latency under 10ms.

**Optimization 3 — BM25 In-Memory Index:**
```python
from rank_bm25 import BM25Okapi
import pickle

# Build BM25 index at startup
tokenized_corpus = [doc.split() for doc in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Query is instantaneous (microseconds for 5000 docs)
scores = bm25.get_scores(query.split())
top_indices = np.argsort(scores)[-20:][::-1]
```

Since the corpus was ~5,000 chunks, BM25 in memory was blazing fast (<5ms).

**Optimization 4 — Cross-Encoder with ONNX Runtime:**
```python
from optimum.onnxruntime import ORTModelForSequenceClassification

# Convert cross-encoder to ONNX for 2-3x speedup
reranker_onnx = ORTModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    export=True
)
# Re-ranking 25 candidates: ~80ms (PyTorch) → ~35ms (ONNX)
```

**Optimization 5 — Parallel Retrieval:**
Dense and sparse retrieval run in parallel using Python's `concurrent.futures`:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    dense_future = executor.submit(dense_retrieve, query_embedding, top_k=20)
    sparse_future = executor.submit(bm25_retrieve, query, top_k=20)
    
    dense_results = dense_future.result()
    sparse_results = sparse_future.result()
```

**Optimization 6 — Caching:**
I implemented an LRU cache for frequent queries:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query_hash):
    return full_retrieval_pipeline(query)
```

Approximately 30% of queries were repeated or near-duplicates, so caching gave a meaningful hit rate.

**Optimization 7 — Pre-warming:** The Flask application pre-loads all models and indexes at startup, not on first request.

---

## Q55. ChromaDB vs FAISS vs Pinecone — you've used ChromaDB and FAISS. When would you pick one over the other?

**Answer:**

I have hands-on experience with both ChromaDB and FAISS, and I have evaluated Pinecone for production use. Here is my detailed comparison.

**ChromaDB:**
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("docs", metadata={"hnsw:space": "cosine"})

# Simple API — add, query, update, delete
collection.add(ids=["doc1"], embeddings=[[0.1, 0.2, ...]], documents=["text"])
results = collection.query(query_embeddings=[[0.1, 0.2, ...]], n_results=5)
```

**When I chose ChromaDB (Knowledge Chatbot):**
- Needed metadata filtering (e.g., `where={"category": "API docs"}`)
- Needed persistence without external infrastructure
- Needed CRUD operations (add/update/delete individual documents)
- Dataset was moderate (~5,000 vectors)
- LangChain integration was seamless

**FAISS:**
```python
import faiss
import numpy as np

# Create index
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)

# For larger datasets, use IVF for speed
nlist = 100
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(vectors)
index.add(vectors)

# Search
distances, indices = index.search(query_vector, k=10)
```

**When I would choose FAISS:**
- Pure similarity search with no metadata filtering needs
- Very large datasets (millions to billions of vectors)
- Maximum search speed is the priority
- GPU acceleration needed (`faiss-gpu`)
- Batch processing scenarios (offline, not real-time)
- No need for CRUD — just build index and search

**Pinecone (Managed Service):**
```python
import pinecone

pinecone.init(api_key="xxx", environment="us-east-1-aws")
index = pinecone.Index("knowledge-base")

index.upsert(vectors=[("doc1", [0.1, 0.2, ...], {"category": "API"})])
results = index.query(vector=[0.1, 0.2, ...], top_k=5, filter={"category": "API"})
```

**When I would choose Pinecone:**
- Team wants zero infrastructure management
- Need distributed, highly available vector search
- Multi-tenant SaaS application
- Budget allows managed service costs
- Need built-in features like namespaces, metadata filtering, and hybrid search

**Decision Matrix:**

| Factor | ChromaDB | FAISS | Pinecone |
|--------|----------|-------|----------|
| Setup Complexity | Low | Medium | Very Low |
| Max Scale | ~1M vectors | Billions | Billions |
| Metadata Filtering | Yes | No (manual) | Yes |
| CRUD Operations | Yes | Limited | Yes |
| Infrastructure | Embedded/Local | Embedded | Managed Cloud |
| Cost | Free | Free | Pay-per-use |
| Latency (1M vectors) | ~10-50ms | ~1-5ms | ~10-30ms |
| GPU Support | No | Yes | N/A |
| LangChain Integration | Native | Native | Native |

**My recommendation pattern:**
- **Prototype/Small scale (<100K vectors):** ChromaDB — fastest to get started, full-featured
- **Large scale, latency-critical:** FAISS — unbeatable raw speed, especially with GPU
- **Production SaaS with operational maturity needs:** Pinecone — no infrastructure burden
- **Hybrid approach:** Use ChromaDB for development and FAISS for production, with a shared abstraction layer

---

## Q56. How does OpenSearch work internally? Explain the inverted index, segments, and the Lucene layer.

**Answer:**

OpenSearch is built on Apache Lucene and inherits its core data structures. Understanding the internals was essential for optimizing the product search at SuperOps.

**The Lucene Layer:**
OpenSearch is essentially a distributed wrapper around Lucene. Each OpenSearch shard is a single Lucene index. When you index a document into OpenSearch, it ultimately reaches a Lucene index writer.

**Inverted Index — The Core Data Structure:**

An inverted index maps terms to the documents containing them, which is the opposite of a forward index (document to terms).

```
Forward Index:
  Doc1 → ["product", "search", "feature"]
  Doc2 → ["search", "engine", "fast"]

Inverted Index:
  "product" → [Doc1]
  "search"  → [Doc1, Doc2]    (with positions: Doc1:pos1, Doc2:pos0)
  "feature" → [Doc1]
  "engine"  → [Doc2]
  "fast"    → [Doc2]
```

Each entry in the inverted index stores:
- **Term dictionary:** Sorted list of all unique terms (stored as an FST — Finite State Transducer for memory efficiency)
- **Postings list:** For each term, a list of document IDs containing that term
- **Term frequency (TF):** How often the term appears in each document
- **Positions:** Where in the document the term appears (enables phrase queries)
- **Offsets:** Character offsets for highlighting

**Segments — Immutable Index Units:**

Lucene does not modify existing index structures. Instead, it uses an append-only, segment-based architecture:

```
Shard (Lucene Index)
├── Segment 0 (committed, immutable)
│   ├── Inverted Index
│   ├── Stored Fields
│   ├── Doc Values (columnar)
│   └── Points (for numeric/geo)
├── Segment 1 (committed, immutable)
├── Segment 2 (committed, immutable)
└── In-Memory Buffer (uncommitted docs)
```

**Write path:**
1. New documents go into an in-memory buffer (and the transaction log / translog for durability).
2. Periodically (every 1 second by default — the "refresh interval"), the buffer is flushed to a new segment on disk. This is when documents become searchable.
3. Segments are immutable once written. Updates and deletes do not modify segments — deletes are tracked in a separate `.del` bitset, and updates are a delete + re-add.

**Segment Merging:**
Over time, many small segments accumulate. OpenSearch runs a background merge process that combines smaller segments into larger ones:
- Reclaims space from deleted documents
- Reduces the number of segments to search (fewer file handles, faster queries)
- Uses a tiered merge policy by default

```
[Seg0: 10 docs] [Seg1: 10 docs] [Seg2: 10 docs]
         ↓ merge
[Seg3: 30 docs (minus deleted)]
```

**Search Path:**
1. Query hits the coordinating node, which fans out to relevant shards.
2. Each shard searches across ALL its segments in parallel.
3. Results from segments are merged within the shard.
4. Shard results are merged at the coordinating node.
5. Final top-k results are returned.

**Key internal structures:**

- **Doc Values:** Column-oriented storage for sorting and aggregations. Instead of reading entire documents, OpenSearch reads just the column needed.
- **Points (BKD-tree):** For numeric and geo-point fields — enables fast range queries.
- **FST (Finite State Transducer):** Compact representation of the term dictionary, kept in memory for fast prefix lookups and fuzzy matching.

Understanding these internals helped me optimize index settings at SuperOps — for example, increasing the refresh interval from 1s to 5s for bulk indexing, and tuning the merge policy for our read-heavy workload.

---

## Q57. What is BM25 scoring? How does it differ from TF-IDF?

**Answer:**

BM25 (Best Matching 25) is the default scoring algorithm in OpenSearch/Elasticsearch/Lucene. It is an evolution of TF-IDF that addresses several of its limitations.

**TF-IDF Recap:**
```
score(q, d) = Σ tf(t, d) × idf(t)

where:
  tf(t, d) = count of term t in document d
  idf(t)   = log(N / df(t))
  N        = total documents
  df(t)    = documents containing term t
```

**Problems with raw TF-IDF:**
1. **Unbounded TF:** If a term appears 100 times vs 10 times, TF-IDF gives 10x the score. But relevance does not scale linearly with frequency — after a point, more occurrences add diminishing value.
2. **No document length normalization:** A 10,000-word document naturally has more term occurrences than a 100-word document, but is not necessarily more relevant.

**BM25 Formula:**
```
score(q, d) = Σ idf(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1 × (1 - b + b × |d|/avgdl)]

where:
  k1   = term frequency saturation parameter (default: 1.2)
  b    = document length normalization (default: 0.75)
  |d|  = length of document d
  avgdl = average document length in the corpus
```

**Key Differences Explained:**

**1. Term Frequency Saturation (k1 parameter):**
```python
# TF-IDF: linear growth
tf_idf_contribution = tf  # 10 occurrences = 10x score of 1 occurrence

# BM25: saturating growth
bm25_contribution = (tf * (k1 + 1)) / (tf + k1)
# At k1=1.2:
#   tf=1  → 0.83
#   tf=5  → 0.97
#   tf=10 → 0.99  (nearly saturated)
#   tf=100 → 0.999 (effectively the same as tf=10)
```

This means BM25 recognizes that 100 mentions of "search" is not meaningfully more relevant than 10 mentions.

**2. Document Length Normalization (b parameter):**
```python
# Length normalization factor
norm = 1 - b + b * (doc_length / avg_doc_length)

# b=0: no length normalization
# b=1: full length normalization (proportional to length ratio)
# b=0.75 (default): moderate normalization
```

A long product description containing "wireless" once is penalized compared to a short product title containing "wireless" once, because the short document is more focused.

**3. IDF Refinement:**
```python
# TF-IDF IDF
idf_tfidf = log(N / df)

# BM25 IDF (OpenSearch implementation)
idf_bm25 = log(1 + (N - df + 0.5) / (df + 0.5))
```

BM25's IDF avoids negative values and has better behavior for very common terms.

**Practical Impact at SuperOps:**
In the product search, BM25's length normalization was crucial. Product names are short (3-5 words), descriptions are long (50-200 words). Without length normalization, description matches would dominate over title matches. BM25 with `b=0.75` naturally balanced this. I further tuned it by boosting the title field:

```json
{
  "query": {
    "multi_match": {
      "query": "network monitoring",
      "fields": ["title^3", "description", "features^2"],
      "type": "best_fields"
    }
  }
}
```

The `^3` boost combined with BM25's length normalization meant a title match for "network monitoring" scored significantly higher than a passing mention in a long description.

---

## Q58. How did you configure the OpenSearch index for product search — analyzers, tokenizers, mappings?

**Answer:**

At SuperOps, the product search index required careful configuration to handle product names, technical terms, and natural language queries effectively.

**Index Mapping:**

```json
{
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "product_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "product_synonyms",
            "english_stemmer",
            "edge_ngram_filter"
          ]
        },
        "search_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "product_synonyms",
            "english_stemmer"
          ]
        },
        "keyword_lowercase": {
          "type": "custom",
          "tokenizer": "keyword",
          "filter": ["lowercase"]
        }
      },
      "filter": {
        "product_synonyms": {
          "type": "synonym",
          "synonyms": [
            "rmm, remote monitoring and management",
            "psa, professional services automation",
            "itsm, it service management",
            "msp, managed service provider"
          ]
        },
        "english_stemmer": {
          "type": "stemmer",
          "language": "english"
        },
        "edge_ngram_filter": {
          "type": "edge_ngram",
          "min_gram": 2,
          "max_gram": 15
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "product_name": {
        "type": "text",
        "analyzer": "product_analyzer",
        "search_analyzer": "search_analyzer",
        "fields": {
          "keyword": {
            "type": "keyword"
          },
          "autocomplete": {
            "type": "text",
            "analyzer": "product_analyzer"
          }
        }
      },
      "description": {
        "type": "text",
        "analyzer": "standard"
      },
      "category": {
        "type": "keyword"
      },
      "features": {
        "type": "text",
        "analyzer": "product_analyzer",
        "search_analyzer": "search_analyzer"
      },
      "tags": {
        "type": "keyword"
      },
      "price": {
        "type": "float"
      },
      "popularity_score": {
        "type": "float"
      },
      "created_at": {
        "type": "date"
      },
      "is_active": {
        "type": "boolean"
      }
    }
  }
}
```

**Key Design Decisions:**

**1. Custom Analyzer Chain:**
The `product_analyzer` pipeline: `standard tokenizer → lowercase → synonyms → stemmer → edge_ngram`.

- **Standard tokenizer:** Splits on whitespace and punctuation, handles most product names well.
- **Synonyms:** Critical for SuperOps domain. Users search "RMM" but the product page might say "Remote Monitoring and Management."
- **Stemmer:** Matches "monitoring" with "monitor", "configured" with "configuration."
- **Edge ngrams at index time only:** Index-time edge ngrams enable prefix matching for autocomplete. The search analyzer omits edge ngrams so a search for "mon" matches indexed n-grams ["mo", "mon", "moni", ...].

**2. Multi-field Mappings:**
`product_name` is indexed three ways:
- `product_name` (text): For full-text search with analysis
- `product_name.keyword` (keyword): For exact match, sorting, and aggregations
- `product_name.autocomplete` (text with edge ngrams): For autocomplete functionality

**3. Shard Configuration:**
2 shards with 1 replica. With our product catalog size (~10K products), a single shard would suffice for search speed, but 2 shards allowed parallel indexing and search across the cluster. The replica ensured high availability.

**4. Keyword Fields for Filtering:**
`category` and `tags` are keyword types for exact-match filtering and aggregations (faceted search). Using keyword instead of text avoids analysis overhead and enables precise filtering:

```json
{
  "query": {
    "bool": {
      "must": {
        "multi_match": {
          "query": "patch management",
          "fields": ["product_name^3", "description", "features^2"]
        }
      },
      "filter": [
        {"term": {"category": "security"}},
        {"term": {"is_active": true}}
      ]
    }
  }
}
```

---

## Q59. Did you implement autocomplete/suggest functionality? How?

**Answer:**

Yes, I implemented autocomplete for the product search at SuperOps using a multi-strategy approach that provided both speed and relevance.

**Strategy 1 — Edge N-gram Based Autocomplete (Primary):**

This was the main approach. At index time, product names are broken into edge n-grams:

```
"Network Monitor" → ["ne", "net", "netw", "netwo", "networ", "network", 
                      "mo", "mon", "moni", "monit", "monito", "monitor"]
```

When a user types "net", it matches the indexed n-gram "net" using a standard text query.

```json
// Index-time: edge_ngram analyzer (see Q58 for full config)
// Search-time: standard analyzer (no edge_ngram)

{
  "query": {
    "bool": {
      "must": {
        "match": {
          "product_name.autocomplete": {
            "query": "net mon",
            "operator": "and"
          }
        }
      },
      "filter": {
        "term": {"is_active": true}
      }
    }
  },
  "size": 7,
  "_source": ["product_name", "category", "product_id"],
  "highlight": {
    "fields": {
      "product_name.autocomplete": {}
    }
  }
}
```

**Why edge n-gram over prefix queries?** Prefix queries (`"prefix": {"product_name": "net"}`) do not use the inverted index efficiently — they scan term dictionaries. Edge n-grams convert prefix matching into standard inverted index lookups, which are much faster.

**Strategy 2 — Completion Suggester (For Pure Speed):**

For instant suggestions (under 5ms), I also set up OpenSearch's completion suggester, which uses an in-memory FST (Finite State Transducer):

```json
// Mapping
{
  "suggest_field": {
    "type": "completion",
    "analyzer": "standard",
    "contexts": [
      {
        "name": "category",
        "type": "category"
      }
    ]
  }
}

// Query
{
  "suggest": {
    "product_suggest": {
      "prefix": "net",
      "completion": {
        "field": "suggest_field",
        "size": 5,
        "fuzzy": {
          "fuzziness": 1
        },
        "contexts": {
          "category": ["monitoring"]
        }
      }
    }
  }
}
```

The completion suggester supports fuzzy matching (handles typos like "netork" matching "network") and context filtering (suggest only products in the user's current category).

**Strategy 3 — Search-as-you-type Field (Hybrid):**

```json
{
  "product_name_sayt": {
    "type": "search_as_you_type",
    "max_shingle_size": 3
  }
}
```

This automatically creates sub-fields (`_2gram`, `_3gram`) for partial matching. Useful for multi-word queries where the user has typed a complete first word and started the second.

**Frontend Integration:**

```javascript
// Debounced autocomplete (300ms delay)
let debounceTimer;
searchInput.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
        if (e.target.value.length >= 2) {
            const suggestions = await fetch(`/api/suggest?q=${e.target.value}`);
            renderSuggestions(await suggestions.json());
        }
    }, 300);
});
```

**Performance Characteristics:**
- Edge n-gram approach: ~10-20ms (inverted index lookup)
- Completion suggester: ~2-5ms (in-memory FST)
- Combined response: Under 25ms end-to-end

I used the completion suggester for the initial keystroke suggestions and fell back to edge n-gram queries when the user typed 3+ characters and needed more contextual results.

---

## Q60. How did you handle relevance tuning? Did you use boosting, function scores, or custom scoring?

**Answer:**

Relevance tuning was an iterative process at SuperOps. I used a combination of field boosting, function scores, and custom scoring to match user expectations.

**1. Field Boosting:**
Product name matches are inherently more relevant than description matches:

```json
{
  "query": {
    "multi_match": {
      "query": "remote monitoring",
      "fields": [
        "product_name^5",
        "product_name.autocomplete^2",
        "features^3",
        "description^1",
        "tags^2"
      ],
      "type": "best_fields",
      "tie_breaker": 0.3
    }
  }
}
```

The `tie_breaker=0.3` means: use the best matching field's score, but add 30% of the scores from other matching fields. This rewards documents that match across multiple fields.

**2. Function Score for Business Logic:**

Raw text relevance is not enough. A product that is popular, recently updated, and highly rated should rank higher than an obscure, outdated match:

```json
{
  "query": {
    "function_score": {
      "query": {
        "multi_match": {
          "query": "monitoring tool",
          "fields": ["product_name^5", "features^3", "description"]
        }
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "popularity_score",
            "modifier": "log1p",
            "factor": 2,
            "missing": 1
          },
          "weight": 3
        },
        {
          "gauss": {
            "created_at": {
              "origin": "now",
              "scale": "90d",
              "offset": "7d",
              "decay": 0.5
            }
          },
          "weight": 1
        },
        {
          "filter": {"term": {"is_featured": true}},
          "weight": 5
        }
      ],
      "boost_mode": "multiply",
      "score_mode": "sum"
    }
  }
}
```

**Breakdown:**
- **`field_value_factor` on popularity:** Products with higher usage/click-through rates get boosted. The `log1p` modifier prevents a single extremely popular product from dominating. `log1p(10) = 2.4` vs `log1p(1000) = 6.9` — a 100x popularity difference becomes only ~3x score difference.
- **`gauss` decay on date:** Newer products get a slight boost that decays over 90 days. This keeps the search results fresh without penalizing evergreen products too harshly.
- **Filter boost for featured products:** Featured products receive a flat weight bonus, but only if they also match the text query (because it is inside `function_score`, not a standalone filter).

**3. Custom Script Scoring (Advanced Cases):**

For complex scoring logic that the built-in functions could not express:

```json
{
  "query": {
    "script_score": {
      "query": {"match": {"description": "monitoring"}},
      "script": {
        "source": """
          double textScore = _score;
          double popularity = doc['popularity_score'].value;
          double recencyBoost = 1.0;
          
          long ageInDays = (params.now - doc['created_at'].value.toInstant().toEpochMilli()) / 86400000L;
          if (ageInDays < 30) recencyBoost = 1.5;
          else if (ageInDays < 90) recencyBoost = 1.2;
          
          return textScore * Math.log1p(popularity) * recencyBoost;
        """,
        "params": {"now": 1680000000000}
      }
    }
  }
}
```

**4. Relevance Testing & Iteration:**
I created a test suite of ~50 representative queries with expected top-3 results:

```python
relevance_tests = [
    {"query": "remote monitoring", "expected_top": ["rmm-pro", "remote-agent"]},
    {"query": "patch management", "expected_top": ["patch-manager", "auto-patch"]},
    {"query": "help desk", "expected_top": ["helpdesk-suite", "ticket-system"]},
]

def run_relevance_tests():
    failures = []
    for test in relevance_tests:
        results = search(test["query"])
        top_ids = [r["product_id"] for r in results[:3]]
        for expected in test["expected_top"]:
            if expected not in top_ids:
                failures.append(f"{test['query']}: missing {expected}")
    return failures
```

This regression suite ran after every scoring change to ensure improvements in one area did not regress another.

---

## Q61. How would you add vector search (k-NN) to OpenSearch alongside BM25? When would you want both?

**Answer:**

OpenSearch supports k-NN (k-Nearest Neighbors) vector search natively since version 1.2, which allows combining semantic vector search with traditional BM25 scoring.

**Setting Up k-NN Index:**

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100,
      "knn.algo_param.ef_construction": 256,
      "knn.algo_param.m": 16
    }
  },
  "mappings": {
    "properties": {
      "product_name": {
        "type": "text",
        "analyzer": "product_analyzer"
      },
      "description": {
        "type": "text"
      },
      "description_vector": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 256,
            "m": 16
          }
        }
      }
    }
  }
}
```

**Indexing with Vectors:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

for product in products:
    vector = model.encode(product["description"]).tolist()
    opensearch_client.index(
        index="products",
        body={
            "product_name": product["name"],
            "description": product["description"],
            "description_vector": vector,
            "category": product["category"]
        }
    )
```

**Hybrid Query — BM25 + k-NN:**

OpenSearch 2.10+ supports the `hybrid` query type natively:

```json
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "description": {
              "query": "tool for monitoring network devices"
            }
          }
        },
        {
          "knn": {
            "description_vector": {
              "vector": [0.12, -0.34, ...],
              "k": 20
            }
          }
        }
      ]
    }
  },
  "search_pipeline": "hybrid_pipeline"
}
```

**Setting up the normalization pipeline:**
```json
PUT /_search/pipeline/hybrid_pipeline
{
  "description": "Hybrid search normalization",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.4, 0.6]
          }
        }
      }
    }
  ]
}
```

This normalizes BM25 and k-NN scores to the same scale, then combines them with configurable weights (here, 60% weight to vector similarity, 40% to BM25).

**When You Want Both (BM25 + k-NN):**

1. **Product search with natural language queries:** Users sometimes search by product name ("SuperOps RMM") where BM25 excels, and sometimes by description ("tool that helps MSPs manage client endpoints remotely") where semantic search excels.

2. **Handling vocabulary mismatch:** A user searching "laptop" should find products described as "notebook computer." BM25 misses this, but vector similarity catches it.

3. **Exact-match + semantic fallback:** For queries containing model numbers or SKUs (e.g., "SUP-MON-2024"), BM25 nails the exact match. For conceptual queries, k-NN provides semantic understanding.

4. **E-commerce search:** Product search benefits heavily from hybrid because users mix exact terms ("32GB RAM") with conceptual needs ("fast gaming laptop").

**When BM25 alone suffices:**
- Structured data with well-defined fields (category, tags, product codes)
- Short, keyword-focused queries
- When you cannot afford the memory overhead of storing vectors

**When k-NN alone suffices:**
- Pure semantic search (Q&A, document retrieval)
- When queries are always natural language
- When the vocabulary gap between queries and documents is large

**Performance consideration:** k-NN adds significant memory overhead (384 dimensions x 4 bytes x number of documents). For 1M products, that is ~1.5GB just for vectors. HNSW graphs add more. I would recommend starting with BM25, measuring where it fails, and adding k-NN selectively for fields that need semantic matching.

---

## Q62. Explain LangChain's architecture — chains, agents, tools, memory. How do they compose?

**Answer:**

LangChain provides a modular framework for building LLM applications. I used it extensively in the RAG Knowledge Chatbot and want to walk through how its core abstractions compose together.

**1. Chains — Sequential Composition:**

A Chain is a sequence of operations where each step's output feeds into the next. The simplest is `LLMChain`:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="Answer based on context:\n{context}\n\nQuestion: {question}"
)

chain = LLMChain(llm=OpenAI(), prompt=prompt)
result = chain.run(question="What is RAG?", context="RAG stands for...")
```

Chains can be composed using `SequentialChain`:
```python
from langchain.chains import SequentialChain

# Chain 1: Rephrase question
rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt, output_key="rephrased")

# Chain 2: Retrieve and answer
answer_chain = LLMChain(llm=llm, prompt=answer_prompt, output_key="answer")

pipeline = SequentialChain(
    chains=[rephrase_chain, answer_chain],
    input_variables=["question"],
    output_variables=["rephrased", "answer"]
)
```

**2. Agents — Dynamic Routing:**

Unlike chains (which follow a fixed sequence), agents use the LLM to decide which actions to take at each step:

```python
from langchain.agents import initialize_agent, Tool, AgentType

tools = [
    Tool(
        name="ProductSearch",
        func=search_products,
        description="Search the product catalog. Input: search query string."
    ),
    Tool(
        name="KnowledgeBase",
        func=rag_retrieve,
        description="Search knowledge base for technical docs. Input: question."
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Perform math calculations. Input: math expression."
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5
)

# The LLM dynamically decides which tool to use
agent.run("How much does the monitoring product cost per 100 endpoints?")
```

The agent follows the ReAct pattern: **Thought → Action → Observation → Thought → ... → Final Answer**.

**3. Tools — External Capabilities:**

Tools are functions that agents can call. Each tool has a name, description (used by the LLM to decide when to call it), and an implementation:

```python
from langchain.tools import BaseTool

class OpenSearchTool(BaseTool):
    name = "product_search"
    description = "Search products by name or description"
    
    def _run(self, query: str) -> str:
        results = opensearch_client.search(index="products", body={
            "query": {"multi_match": {"query": query, "fields": ["name", "description"]}}
        })
        return format_results(results)
    
    def _arun(self, query: str) -> str:
        # Async version
        pass
```

**4. Memory — Conversational State:**

Memory allows chains/agents to maintain state across interactions:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # Keep last 5 exchanges
    memory_key="chat_history",
    return_messages=True
)

# Different memory types for different needs:
# ConversationBufferMemory — stores everything (risky for long conversations)
# ConversationBufferWindowMemory — sliding window of last k exchanges
# ConversationSummaryMemory — LLM summarizes older messages
# ConversationSummaryBufferMemory — recent messages verbatim + summary of older ones
```

**How They Compose in My RAG Pipeline:**

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Compose: Embeddings + VectorStore + Retriever + LLM + Memory
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",  # stuff all docs into context
    retriever=retriever,
    memory=ConversationBufferWindowMemory(k=3),
    return_source_documents=True
)

result = qa_chain({"query": "How do I configure alerts?"})
```

**LCEL (LangChain Expression Language) — Modern Approach:**
LangChain has moved toward a more composable pipe syntax:

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)
```

This is more Pythonic and easier to debug than the older Chain classes.

---

## Q63. What is the difference between a Chain and an Agent in LangChain?

**Answer:**

This is a fundamental architectural distinction in LangChain that dictates how your application makes decisions.

**Chain — Deterministic, Pre-defined Flow:**

A Chain executes a fixed sequence of steps defined at build time. The execution path is always the same regardless of the input:

```python
# Chain: ALWAYS does Step A → Step B → Step C
# The LLM generates text, but doesn't decide what to do next

from langchain.chains import SequentialChain

# Step 1: Always rephrase the question
rephrase = LLMChain(llm=llm, prompt=rephrase_prompt, output_key="clean_query")

# Step 2: Always retrieve documents
retrieve = LLMChain(llm=llm, prompt=retrieve_prompt, output_key="context")

# Step 3: Always generate answer
answer = LLMChain(llm=llm, prompt=answer_prompt, output_key="answer")

pipeline = SequentialChain(chains=[rephrase, retrieve, answer])
# Every query follows the same 3-step path
```

**Characteristics of Chains:**
- Predictable execution — you know exactly what will happen
- Easier to debug and test
- Lower latency (no decision overhead)
- Lower cost (fewer LLM calls)
- Cannot adapt to unexpected inputs

**Agent — Dynamic, LLM-Driven Routing:**

An Agent uses the LLM itself to decide which actions to take, in what order, and when to stop:

```python
# Agent: LLM decides at each step what to do next
# Could call ToolA, then ToolB, then ToolA again, then answer
# Or could skip straight to answering if it already knows

from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[search_tool, calculator_tool, database_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5
)

# For "What is 2+2?" → Agent might skip tools, answer directly
# For "What's the total revenue from product X?" → 
#   Agent calls database_tool, then calculator_tool, then answers
```

**Execution trace of an Agent:**
```
Query: "How much would 50 licenses of the Enterprise plan cost with the 20% discount?"

Thought: I need to find the Enterprise plan price first.
Action: ProductSearch("Enterprise plan pricing")
Observation: Enterprise plan is $99/month per license.

Thought: Now I need to calculate the total with discount.
Action: Calculator("50 * 99 * 0.8")
Observation: 3960

Thought: I have all the information needed.
Final Answer: 50 Enterprise licenses with 20% discount would cost $3,960/month.
```

**Characteristics of Agents:**
- Flexible — adapts behavior to input
- Can handle multi-step, complex queries
- Higher latency (multiple LLM calls for reasoning)
- Higher cost (each thought/action cycle is an LLM call)
- Harder to debug (non-deterministic paths)
- Risk of infinite loops (need max_iterations guard)

**When I Used Each:**

| Use Case | Choice | Why |
|----------|--------|-----|
| RAG Knowledge Chatbot | Chain | Fixed retrieval pipeline — always retrieve, always answer. Predictable and fast. |
| Complex support queries | Agent | User might need product info, account lookup, and calculation — agent routes dynamically. |
| Batch document processing | Chain | Same steps for every document. |
| Interactive debugging assistant | Agent | Needs to decide whether to search logs, read code, or query monitoring tools. |

**My rule of thumb:** Start with a Chain. Only move to an Agent when the decision of *what to do* genuinely depends on the query content and cannot be predetermined. Agents are powerful but introduce non-determinism, higher latency, and cost that are unjustified if a fixed pipeline works.

**Hybrid approach I often used:** A Chain that contains an Agent at one step:

```python
# Fixed chain, but one step uses agent-like routing
class SmartRetrievalChain:
    def run(self, query):
        # Step 1 (fixed): Classify query type
        query_type = self.classifier.predict(query)
        
        # Step 2 (dynamic): Route based on classification
        if query_type == "factual":
            docs = self.rag_retriever.get(query)
        elif query_type == "product":
            docs = self.product_search.get(query)
        elif query_type == "account":
            docs = self.account_lookup.get(query)
        
        # Step 3 (fixed): Generate answer
        return self.llm.generate(docs, query)
```

This gives you the predictability of chains with selective dynamism where needed.

---

## Q64. What is function calling in LLMs? How does it work under the hood (OpenAI's implementation)?

**Answer:**

Function calling (now called "tool use" in newer API versions) is a mechanism that allows LLMs to output structured JSON representing a function invocation instead of (or alongside) natural language text. It bridges the gap between free-form language models and structured software systems.

**The Problem It Solves:**
Without function calling, you had to prompt-engineer the LLM to output JSON and then parse it — which was fragile:
```
# Old approach (unreliable)
prompt = "Extract the city and date. Output JSON: {city: ..., date: ...}"
response = "Sure! Here's the JSON: {city: 'Paris', date: '2024-01-15'}" 
# Now you have to parse free text to find the JSON — error-prone
```

**How It Works (OpenAI's Implementation):**

**Step 1 — Define functions in the API call:**
```python
import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather in Chennai?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ],
    tool_choice="auto"  # "auto", "none", or specific function
)
```

**Step 2 — Model returns a tool call (not plain text):**
```python
# response.choices[0].message:
{
    "role": "assistant",
    "content": null,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\": \"Chennai\", \"unit\": \"celsius\"}"
            }
        }
    ]
}
```

The model does NOT execute the function. It outputs structured JSON saying "call this function with these arguments." Your application code executes it.

**Step 3 — Execute and feed result back:**
```python
# Your code calls the actual function
weather_data = get_weather(city="Chennai", unit="celsius")

# Feed the result back to the model
messages.append(response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": json.dumps(weather_data)
})

# Model generates final response incorporating the tool result
final_response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)
# "The current weather in Chennai is 32°C with high humidity."
```

**Under the Hood — How the Model Learns This:**

1. **Training data:** The model is fine-tuned on examples of function definitions + user queries + correct function call outputs. This teaches it to map natural language intent to structured function invocations.

2. **System prompt injection:** The function definitions are serialized (likely as a special format in the system prompt) that the model has been trained to recognize and respond to with structured output.

3. **Constrained decoding:** When the model decides to make a function call, the decoding process constrains output to valid JSON matching the function schema. This is why function calling almost never produces malformed JSON, unlike free-text prompting.

4. **`tool_choice` parameter:**
   - `"auto"`: Model decides whether to call a function or respond with text
   - `"none"`: Model must respond with text only
   - `{"type": "function", "function": {"name": "get_weather"}}`: Force a specific function call

**How I Used Function Calling:**

In the RAG chatbot, I used function calling to structure the LLM's retrieval decisions:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for relevant documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "enum": ["api", "setup", "troubleshooting"]},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]
```

This was more reliable than having the LLM output a search query in free text, because function calling guaranteed structured, parseable output every time.

---

## Q65. Explain prompt engineering techniques you've used — few-shot, chain-of-thought, self-consistency, ReACT.

**Answer:**

I have applied all four of these techniques in production systems. Each addresses a different failure mode of LLM reasoning.

**1. Few-Shot Prompting:**

Providing examples in the prompt to demonstrate the desired output format and reasoning pattern:

```python
few_shot_prompt = """You are a product categorizer.

Example 1:
Product: "Wireless noise-cancelling headphones with 30hr battery"
Category: Electronics > Audio > Headphones
Reasoning: The product is a headphone with wireless and battery features.

Example 2:
Product: "Organic cold-pressed coconut oil 500ml"
Category: Grocery > Cooking Oils > Coconut Oil
Reasoning: This is a food/cooking product made from coconut.

Now categorize:
Product: "{product_description}"
Category:
Reasoning:"""
```

**When I used it:** For classification tasks in the chatbot where the LLM needed to understand the output taxonomy. 3-5 examples were usually sufficient; beyond that, diminishing returns and token waste.

**Key insight:** Example selection matters enormously. I dynamically selected few-shot examples that were semantically similar to the input query (using embedding similarity), rather than using static examples.

**2. Chain-of-Thought (CoT) Prompting:**

Instructing the model to reason step by step before giving the final answer:

```python
cot_prompt = """Given the user's question and the retrieved context, 
answer step by step:

Context: {context}
Question: {question}

Let's think step by step:
1. What key information is in the context?
2. How does it relate to the question?
3. What is the answer?

Step-by-step reasoning:"""
```

**When I used it:** For complex queries in the knowledge chatbot that required synthesizing information from multiple chunks. For example, "What are the differences between Plan A and Plan B?" requires the model to identify features in both plans and compare them.

**Impact:** CoT reduced factual errors by approximately 30% on multi-hop questions in my evaluation. The cost is more output tokens (higher latency and cost), so I only activated CoT when the query classifier flagged a question as complex.

**3. Self-Consistency:**

Generate multiple CoT responses and take the majority answer:

```python
def self_consistent_answer(question, context, n=5):
    """Generate n answers and return the most consistent one."""
    answers = []
    for _ in range(n):
        response = llm.generate(
            prompt=cot_prompt.format(question=question, context=context),
            temperature=0.7  # Need some randomness for diverse paths
        )
        final_answer = extract_answer(response)
        answers.append(final_answer)
    
    # Majority vote (or semantic clustering for open-ended answers)
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

**When I used it:** For high-stakes answers where accuracy mattered more than latency (e.g., compliance or configuration recommendations). It is expensive (n times the cost), so I used it selectively.

**Practical variation:** Instead of full self-consistency, I used `n=3` with temperature=0.3, which was a pragmatic middle ground. If all 3 agreed, high confidence. If they diverged, I flagged the response for human review.

**4. ReACT (Reasoning + Acting):**

This is the pattern used by LangChain agents. The model alternates between reasoning (Thought) and action (Tool use):

```python
react_prompt = """Answer the following question using the available tools.

Tools:
- search(query): Search the knowledge base
- calculate(expression): Perform calculations
- lookup_account(id): Get account details

Format:
Thought: [your reasoning about what to do next]
Action: [tool_name(arguments)]
Observation: [result from tool]
... (repeat as needed)
Thought: I now have enough information.
Final Answer: [your answer]

Question: {question}

Thought:"""
```

**When I used it:** For the agent-based components of the chatbot where the query required multiple information sources. For example, "How many endpoints can I monitor on my current plan and how much would an upgrade cost?" requires: (1) look up the user's plan, (2) find plan limits, (3) find upgrade pricing.

**ReACT vs plain CoT:** ReACT is CoT with the ability to take actions in the real world (call APIs, search databases). It is strictly more powerful but also more expensive and harder to control. I bounded it with `max_iterations=5` to prevent runaway loops.

**Technique Selection Matrix:**

| Technique | Best For | Cost | Latency |
|-----------|----------|------|---------|
| Few-shot | Format/classification tasks | Low (static examples) | Low |
| CoT | Complex reasoning | Medium (+50% tokens) | Medium |
| Self-consistency | High-stakes accuracy | High (n times cost) | High |
| ReACT | Multi-step, tool-using tasks | High (multiple LLM calls) | High |

---

## Q66. How do you handle context window limits when the input exceeds the model's token limit?

**Answer:**

Context window management was a constant challenge in the RAG chatbot, especially when multiple retrieved chunks plus conversation history approached the token limit. I implemented several strategies.

**Strategy 1 — Smart Chunk Selection (Most Common):**

Rather than stuffing all retrieved chunks into the context, I select only the most relevant ones that fit within the budget:

```python
import tiktoken

def fit_chunks_to_context(chunks, query, max_context_tokens=3000, model="gpt-3.5-turbo"):
    """Select chunks that fit within token budget, prioritized by relevance."""
    encoder = tiktoken.encoding_for_model(model)
    
    selected = []
    current_tokens = 0
    
    # Chunks are already sorted by relevance (from re-ranker)
    for chunk in chunks:
        chunk_tokens = len(encoder.encode(chunk.page_content))
        if current_tokens + chunk_tokens <= max_context_tokens:
            selected.append(chunk)
            current_tokens += chunk_tokens
        else:
            break  # Stop adding once budget is exhausted
    
    return selected
```

**Token Budget Allocation:**
```
GPT-3.5-turbo context: 4096 tokens (or 16K variant)
├── System prompt:        ~200 tokens
├── Conversation history: ~500 tokens (last 2-3 exchanges)
├── Retrieved context:    ~2500 tokens (4-6 chunks)
├── User query:           ~50 tokens
└── Response budget:      ~800 tokens
```

**Strategy 2 — Map-Reduce for Large Document Sets:**

When the answer requires information scattered across many documents that cannot all fit:

```python
from langchain.chains.summarize import load_summarize_chain

# Map: Summarize/extract from each chunk independently
# Reduce: Combine the summaries into a final answer

chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,    # "Extract relevant info about {question} from: {text}"
    combine_prompt=reduce_prompt  # "Combine these extracts into a final answer: {text}"
)
```

This processes chunks in parallel (map step), then combines the results (reduce step). Each individual LLM call stays within the context limit.

**Strategy 3 — Conversation History Compression:**

For multi-turn conversations, older messages are summarized:

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,  # Keep last N tokens verbatim
    # Older messages get summarized into a condensed form
)

# Turn 1-5: Summarized as "User asked about monitoring setup and was told to..."
# Turn 6-8: Kept verbatim as recent context
```

**Strategy 4 — Recursive Summarization:**

For very long documents (e.g., 50-page technical guides) that need to be fully processed:

```python
def recursive_summarize(text, max_tokens=4000, chunk_size=3000):
    chunks = split_text(text, chunk_size)
    
    while len(chunks) > 1:
        summaries = []
        for chunk in chunks:
            summary = llm.generate(f"Summarize concisely:\n{chunk}")
            summaries.append(summary)
        
        # Combine summaries and re-chunk if still too large
        combined = "\n".join(summaries)
        if count_tokens(combined) <= max_tokens:
            return combined
        chunks = split_text(combined, chunk_size)
    
    return chunks[0]
```

**Strategy 5 — Selecting the Right Model:**

Sometimes the simplest strategy is using a model with a larger context window:

| Model | Context Window | When I Used |
|-------|---------------|-------------|
| GPT-3.5-turbo | 4K / 16K | Standard queries (fast, cheap) |
| GPT-4 | 8K / 32K | Complex queries needing more context |
| GPT-4-turbo | 128K | Rare cases with massive context needs |
| Claude | 100K-200K | Document analysis tasks |

**My dynamic approach:** The system estimates required context size based on the number of relevant chunks found, and routes to the appropriate model:

```python
def select_model(query, retrieved_chunks):
    total_tokens = sum(count_tokens(c.page_content) for c in retrieved_chunks)
    
    if total_tokens < 3000:
        return "gpt-3.5-turbo"  # Fast, cheap
    elif total_tokens < 7000:
        return "gpt-3.5-turbo-16k"
    elif total_tokens < 25000:
        return "gpt-4-turbo"
    else:
        return "gpt-4-turbo"  # With map-reduce fallback
```

This balances cost and quality — 80% of queries used the cheapest model because the context fit comfortably.

---

## Q67. What is LangFuse? How did you use it for LLM observability?

**Answer:**

LangFuse is an open-source LLM observability and analytics platform. I used it to monitor, debug, and improve the RAG Knowledge Chatbot in production.

**Why LangFuse Over Alternatives:**
- Open source (self-hostable) — no data leaves our infrastructure
- Native LangChain integration (one-line setup)
- Traces, spans, and generations with full hierarchy
- Cost tracking per request
- Prompt versioning and management
- Evaluation scoring (human and automated)

**Integration Setup:**

```python
from langfuse.callback import CallbackHandler

# Initialize LangFuse handler
langfuse_handler = CallbackHandler(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://langfuse.our-domain.com"  # Self-hosted
)

# Attach to LangChain — every chain/agent call is automatically traced
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    callbacks=[langfuse_handler]
)

result = qa_chain({"query": "How to set up alerts?"})
```

That single callback handler captures the entire execution trace automatically.

**What I Tracked:**

**1. Trace-Level Metrics (Per Request):**
```python
# Manual trace for custom pipeline steps
from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(
    name="rag_query",
    user_id=user_id,
    metadata={"source": "chatbot_v2"}
)

# Span for retrieval
retrieval_span = trace.span(
    name="hybrid_retrieval",
    input={"query": query}
)
# ... do retrieval ...
retrieval_span.end(output={
    "num_results": len(results),
    "top_score": results[0].score,
    "latency_ms": retrieval_latency
})

# Generation for LLM call
generation = trace.generation(
    name="answer_generation",
    model="gpt-3.5-turbo",
    input=[{"role": "user", "content": prompt}],
    output=response.content,
    usage={
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }
)
```

**2. Cost Tracking:**
LangFuse automatically calculates cost per request based on model and token usage. I set up alerts when daily costs exceeded thresholds:
- Average cost per query: ~$0.003 (GPT-3.5-turbo)
- Daily budget alert: $50
- Per-query cost spike alert: >$0.05 (indicates excessive retries or large context)

**3. Prompt Management & Versioning:**

```python
# Store prompts in LangFuse with versions
prompt = langfuse.get_prompt("rag_system_prompt", version=3)

# Use in chain
system_message = prompt.compile(context=context, question=question)
```

This allowed me to A/B test prompt versions. When I changed the system prompt, I could compare quality metrics between v2 and v3 side by side in the LangFuse dashboard.

**4. Evaluation Scores:**

```python
# Automated evaluation after each response
trace.score(
    name="retrieval_relevance",
    value=compute_relevance_score(query, retrieved_chunks),
    comment="Cosine similarity between query and top chunk"
)

trace.score(
    name="answer_faithfulness",
    value=check_faithfulness(response, retrieved_chunks),
    comment="Does answer stay within retrieved context?"
)

# Human feedback (from thumbs up/down in the UI)
langfuse.score(
    trace_id=trace.id,
    name="user_feedback",
    value=1,  # 1 = positive, 0 = negative
    comment="User clicked thumbs up"
)
```

**5. Dashboard and Alerts:**

I set up dashboards tracking:
- P50/P95/P99 latency across the pipeline
- Retrieval relevance scores distribution
- Hallucination rate (faithfulness < 0.5)
- User satisfaction (feedback scores)
- Token usage and cost trends
- Error rates by query category

**Production Debugging Example:**
A user reported "the chatbot gives wrong answers about pricing." Using LangFuse, I:
1. Filtered traces by user ID
2. Found the specific trace with low faithfulness score
3. Saw that retrieval returned outdated pricing chunks (document hadn't been re-indexed)
4. Fixed the incremental indexing pipeline for pricing documents
5. Verified improvement by comparing faithfulness scores before/after

LangFuse turned LLM debugging from "why did it say that?" (black box) into a fully traceable pipeline where I could pinpoint exactly which stage (retrieval, re-ranking, generation) caused the issue.

---

## Q68. How do you evaluate LLM outputs in production? What metrics and frameworks did you use?

**Answer:**

Evaluating LLM outputs is fundamentally harder than evaluating traditional ML models because outputs are free-form text. I built a multi-layered evaluation system combining automated metrics, LLM-as-judge, and human evaluation.

**Layer 1 — Automated Retrieval Metrics (Fast, Cheap):**

These evaluate the retrieval component, not the LLM itself:

```python
# Computed on every request, logged to LangFuse
metrics = {
    "retrieval_latency_ms": 45,
    "top_chunk_similarity": 0.82,
    "num_chunks_above_threshold": 4,  # chunks with similarity > 0.3
    "context_token_count": 1850,
}
```

**Layer 2 — LLM-as-Judge (Automated Quality Scoring):**

I used a separate LLM call to evaluate the primary LLM's output across multiple dimensions:

```python
evaluation_prompt = """You are an expert evaluator. Rate the following response on a scale of 1-5 for each criterion.

Question: {question}
Context provided: {context}
Response: {response}

Rate each:
1. Faithfulness (1-5): Does the response ONLY use information from the provided context? 
   5 = perfectly faithful, 1 = contains fabricated information
2. Relevance (1-5): Does the response actually answer the question?
   5 = directly answers, 1 = completely off-topic  
3. Completeness (1-5): Does the response cover all aspects of the question?
   5 = comprehensive, 1 = misses key information
4. Coherence (1-5): Is the response well-structured and easy to understand?
   5 = crystal clear, 1 = confusing/contradictory

Output JSON: {{"faithfulness": X, "relevance": X, "completeness": X, "coherence": X}}"""

def evaluate_response(question, context, response):
    eval_result = eval_llm.generate(
        evaluation_prompt.format(
            question=question, context=context, response=response
        )
    )
    scores = json.loads(eval_result)
    return scores
```

I ran this on a sample (10-20%) of production queries to keep costs manageable.

**Layer 3 — RAGAS Framework:**

I used the RAGAS (Retrieval Augmented Generation Assessment) framework for systematic evaluation:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

eval_data = Dataset.from_dict({
    "question": questions,
    "answer": generated_answers,
    "contexts": retrieved_contexts,
    "ground_truth": reference_answers,  # For metrics that need ground truth
})

results = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=eval_llm
)

print(results)
# {'faithfulness': 0.87, 'answer_relevancy': 0.82, 
#  'context_precision': 0.79, 'context_recall': 0.84}
```

**RAGAS metrics explained:**
- **Faithfulness:** Claims in the answer that can be traced back to the context. Catches hallucination.
- **Answer Relevancy:** Is the answer pertinent to the question? (Reverse-generates questions from the answer and checks similarity to original.)
- **Context Precision:** Are the top-ranked retrieved chunks actually relevant? (Checks if relevant chunks are ranked higher.)
- **Context Recall:** Are all pieces of information needed to answer the question present in the context?

**Layer 4 — Human Evaluation (Weekly Sampling):**

Every week, I sampled 20-30 production queries and had team members rate them:

```python
human_eval_template = {
    "query": "...",
    "response": "...",
    "retrieved_chunks": ["...", "..."],
    "ratings": {
        "correctness": None,       # 1-5: Is the answer factually correct?
        "helpfulness": None,       # 1-5: Would this help the user?
        "safety": None,            # Pass/Fail: Any harmful content?
        "source_attribution": None # Pass/Fail: Are sources properly cited?
    },
    "notes": ""
}
```

**Layer 5 — User Feedback (Continuous Signal):**

The chatbot UI had thumbs up/down buttons. This implicit signal was the most valuable long-term metric:

```python
# Track feedback rate and satisfaction
feedback_metrics = {
    "total_queries": 1000,
    "queries_with_feedback": 350,   # 35% feedback rate
    "positive_feedback": 290,        # 82.8% satisfaction
    "negative_feedback": 60,
}
```

Negative feedback triggers were automatically flagged for review in LangFuse.

**Layer 6 — Regression Testing (Pre-Deployment):**

Before deploying any changes (new prompt, different model, retrieval tuning), I ran a regression test:

```python
golden_test_set = [
    {
        "query": "How to reset my password?",
        "expected_contains": ["settings", "account", "reset link"],
        "expected_not_contains": ["credit card", "billing"],
        "min_faithfulness": 0.8,
    },
    # ... 100+ test cases
]

def run_regression(pipeline):
    results = []
    for test in golden_test_set:
        response = pipeline.query(test["query"])
        passed = all([
            all(kw in response.lower() for kw in test["expected_contains"]),
            all(kw not in response.lower() for kw in test["expected_not_contains"]),
            evaluate_faithfulness(response) >= test["min_faithfulness"],
        ])
        results.append({"test": test["query"], "passed": passed})
    
    pass_rate = sum(r["passed"] for r in results) / len(results)
    return pass_rate, results
```

**Production Monitoring Dashboard Summary:**

| Metric | Target | Actual (Avg) | Alert Threshold |
|--------|--------|--------------|-----------------|
| Faithfulness | > 0.85 | 0.87 | < 0.75 |
| Answer Relevancy | > 0.80 | 0.82 | < 0.70 |
| User Satisfaction | > 80% | 82.8% | < 70% |
| P95 Latency (retrieval) | < 200ms | 165ms | > 300ms |
| Hallucination Rate | < 5% | 3.2% | > 10% |
| Cost Per Query | < $0.005 | $0.003 | > $0.01 |

The combination of automated metrics (fast feedback loop), LLM-as-judge (scalable quality assessment), and human evaluation (ground truth calibration) gave me comprehensive coverage. Automated metrics caught regressions quickly; human evaluation ensured the automated metrics themselves remained calibrated.
