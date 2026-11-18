"""
Central configuration for the Multimodal Stateful Hybrid RAG System.
The only paid-capable service used here is Gemini, but it's on the FREE tier
(no credit card, free daily quota). Everything else runs locally.
"""

# ---- LLM Settings (the ONE and ONLY LLM used in this project) ----
GEMINI_API_KEY = "--- YOUR API KEY ---"
GEMINI_MODEL = "----"   # free tier model

# ---- Embedding model (runs locally, free, supports English + Tamil) ----
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ---- spaCy model for entity extraction (Knowledge Graph) ----
SPACY_MODEL = "en_core_web_sm"

# ---- Storage paths (local, free) ----
KG_STORE_PATH = "kg_store/graph.pkl"
CHROMA_DB_PATH = "chroma_store"
DOCS_FOLDER = "uploaded_docs"

# ---- Chunking settings ----
CHUNK_SIZE = 200       # characters per chunk
CHUNK_OVERLAP = 40

# ---- Hybrid Search settings ----
TOP_K = 5              # results to retrieve from each method before fusion
RRF_K = 60             # standard constant used in Reciprocal Rank Fusion