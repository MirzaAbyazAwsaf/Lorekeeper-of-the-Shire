# Project Code Documentation

This document explains every file and piece of code in the Tolkien RAG chatbot project.

## Project Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about J.R.R. Tolkien's works using a local vector database and LLM.

**Files:**
| File | Purpose |
|------|---------|
| `main.py` | Interactive CLI chatbot (user interface + prompt + LLM call) |
| `vector_db.py` | Document ingestion, vector store, and retrieval logic |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |
| `data/` | Source PDF documents (9 files) |
| `chroma_db/` | Persisted vector store (SQLite) |

---

## 1. main.py — The Chatbot

### Imports (lines 1-3)
```python
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_db import get_retriever
```
- `OllamaLLM` — interface to run the local language model through Ollama
- `ChatPromptTemplate` — template-based prompt builder from LangChain
- `get_retriever` — imported from `vector_db.py`; returns the retrieval pipeline

### `format_passages(passages)` (lines 6-13)
```python
def format_passages(passages):
    parts = []
    for p in passages:
        book = p.metadata.get("book", "Unknown")
        chapter = p.metadata.get("chapter", p.metadata.get("section", ""))
        header = f"[{book}-{chapter}]" if chapter else f"[{book}]"
        parts.append(f"{header}\n{p.page_content}")
    return "\n\n---\n\n".join(parts)
```
**What it does:**
- Takes the list of retrieved documents (each is a `Document` object)
- Extracts `book` and `chapter`/`section` from each document's metadata
- Builds a header like `[The Hobbit-Chapter 1: An Unexpected Party]`
- Returns all passages joined by `---` separators so the LLM can clearly see citation source per passage

### `main()` (lines 16-77)
**Getting the retriever (line 17):**
```python
retriever = get_retriever()
```
Returns the hybrid retriever with reranking (defined in `vector_db.py`).

**Initializing the model (line 18):**
```python
model = OllamaLLM(model="qwen2.5:7b-instruct-q4_K_M", client_kwargs={"trust_env": False})
```
- Uses Qwen2.5 7B (instruct, quantized q4_K_M) running locally via Ollama
- `trust_env: False` — prevents Ollama from reading proxy/env settings

**System prompt template (lines 20-57):**
The prompt defines:
- The assistant's persona (Tolkien scholar/lore-master)
- Instruction: if `{passages}` is empty, the question is NOT about Tolkien — respond naturally
- Instruction: if passages exist, answer ONLY using them; never fabricate
- Answering style (cite book/chapter, direct quotes, thorough detail)
- Special formatting rules for genealogy/family tree questions (vertical `|` for parent-child, `.....` for spouses)
- Example family tree format

**Building the chain (lines 59-60):**
```python
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
```
- `|` operator pipes prompt output into the model (LangChain LCEL)

**Banner (lines 62-65):**
Prints the title bar with quit instructions.

**Main loop (lines 67-77):**
```python
while True:
    question = input("\n> ")
    if question.strip().lower() == "q":
        print("Mellon nath, farewell!")
        break
    if not question.strip():
        continue
    docs = retriever.invoke(question)
    results = chain.invoke({"passages": format_passages(docs), "question": question})
    print(f"\n{results}")
```
- `q` quits the loop
- Blank input is skipped
- For each question:
  1. `retriever.invoke(question)` — runs hybrid search + rerank + threshold filter
  2. `format_passages(docs)` — turns retrieved docs into text with headers
  3. `chain.invoke({...})` — sends passages + question to the LLM
  4. Prints the answer

---

## 2. vector_db.py — Ingestion & Retrieval

### Imports (lines 1-13)
- `OllamaEmbeddings` — produces embeddings via Ollama (`mxbai-embed-large`)
- `Chroma` — vector store persistence
- `Document` — LangChain document object (content + metadata)
- `BaseRetriever` — base class for custom retrievers
- `RecursiveCharacterTextSplitter` — text chunker
- `BM25Okapi` — keyword-based scoring from `rank_bm25`
- `pymupdf` — PDF text extraction
- `cohere` — Cohere reranking API
- `glob`, `os`, `re`, `uuid`, `nltk` — standard utilities

### Constants (lines 16-20)
```python
BOOK_NAME_MAP = {"the_hobbit": "The Hobbit"}
CHAPTER_RE = re.compile(r"Chapter\s+\d+", re.IGNORECASE)
```
- `BOOK_NAME_MAP` — maps filename → display name (currently only the Hobbit)
- `CHAPTER_RE` — regex to detect `Chapter N` headings

### `BM25Retriever` class (lines 23-34)
```python
class BM25Retriever(BaseRetriever):
    bm25: BM25Okapi
    documents: list
    k: int = 6

    def _get_relevant_documents(self, query: str) -> list:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
        return [self.documents[i] for i in top_k_idx]
```
**What it does:**
- Wraps the BM25Okapi algorithm into a LangChain retriever
- `_get_relevant_documents` is called by `invoke()`
- Tokenizes the query (lowercase, whitespace split)
- BM25 scores each stored document against the query
- Sorts by score descending, returns top `k` (default 6)

**Why BM25:** keyword/exact-match search. Catches precise names like "Finwë" that semantic search may miss.

### `HybridRetriever` class (lines 37-69)
Fields:
- `chroma_retriever` — semantic search (meaning-based)
- `bm25_retriever` — keyword search
- `chroma_weight: 0.6` — 60% weight for semantic
- `bm25_weight: 0.4` — 40% weight for keyword
- `k: 6` — number of results

Method:
```python
def _get_relevant_documents(self, query: str) -> list:
    chroma_docs = self.chroma_retriever.invoke(query)
    bm25_docs = self.bm25_retriever.invoke(query)

    seen = set()
    scored = []

    for doc in chroma_docs:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            scored.append((content, doc.metadata, self.chroma_weight))

    for doc in bm25_docs:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            scored.append((content, doc.metadata, self.bm25_weight))
        else:
            for i, (c, m, s) in enumerate(scored):
                if c == content:
                    scored[i] = (c, m, s + self.bm25_weight)
                    break

    scored.sort(key=lambda x: x[2], reverse=True)
    return [Document(page_content=c, metadata=m) for c, m, s in scored[:self.k]]
```
**What it does:**
1. Runs both retrievers in parallel
2. Merges results; documents found by **both** methods get a boosted score (weight added twice)
3. Sorts by combined score
4. Returns top `k` documents

### `CohereReranker` class (lines 72-99)
**`__init__` (lines 73-76):**
- Creates a Cohere client using the API key
- Default model: `rerank-multilingual-v3.0`
- `top_n`: how many reranked results to return

**`rerank` (lines 78-99):**
```python
def rerank(self, query: str, documents: list) -> list:
    if not documents:
        return []
    docs_text = [doc.page_content for doc in documents]
    response = self.client.rerank(
        query=query,
        documents=docs_text,
        model=self.model,
        top_n=min(self.top_n, len(documents)),
    )
    reranked = []
    for result in response.results:
        doc = documents[result.index]
        reranked.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": result.relevance_score},
            )
        )
    return reranked
```
**What it does:**
- Sends query + candidate documents to Cohere's API
- Cohere returns reordered results with a `relevance_score` per document
- The score is stored in each document's metadata for later filtering
- Returns documents in new relevance order

### `extract_book_name(file_path)` (lines 102-107)
```python
def extract_book_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0].lower()
    base = re.sub(r"[\s_\-]+", "_", base)
    if base in BOOK_NAME_MAP:
        return BOOK_NAME_MAP[base]
    return os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")
```
- Takes a PDF path, gets the filename without extension (lowercased)
- Normalizes underscores/spaces/dashes
- Uses `BOOK_NAME_MAP` if known, otherwise uses the cleaned filename as the book name

### `split_by_chapters(text, source, book_name)` (lines 110-138)
```python
chapters = re.split(r"(?=Chapter\s+\d+)", text, flags=re.IGNORECASE)
for block in chapters:
    ...
    chapter_match = re.match(r"(Chapter\s+\d+)[:\.\s–—-]*(.*)", block, re.IGNORECASE)
    ...
    documents.append(Document(page_content=block, metadata={source, book, chapter}))
```
- Uses a lookahead regex to split text before each `Chapter N` marker
- Extracts chapter number and title via `re.match`
- Creates one `Document` per chapter with metadata: `source` (file path), `book`, `chapter`

### `split_by_sections(text, source, book_name)` (lines 141-159)
```python
sections = text.split(section_marker)  # marker = "==="
for i, section in enumerate(sections):
    ...
    first_line = section.split("\n", 1)[0].strip()
    documents.append(Document(... metadata={"section": first_line[:100]}))
```
- Splits text by `===` separators (documents that use that marker)
- Uses the first line of each section (up to 100 chars) as the section label

### `smart_split(text, source, book_name)` (lines 162-172)
```python
def smart_split(text, source, book_name):
    if book_name == "The Hobbit" and re.search(CHAPTER_RE, text):
        return split_by_chapters(text, source, book_name)
    if "===" in text:
        return split_by_sections(text, source, book_name)
    return [Document(page_content=text.strip(), metadata={...})]
```
- Dispatches to the appropriate splitter:
  1. Hobbit with `Chapter N` → chapter split
  2. Text with `===` → section split
  3. Otherwise → whole document as one chunk

### `load_documents()` (lines 175-207)
```python
db_location = "./chroma_db"
add_docs = not os.path.exists(db_location)
if not add_docs:
    return [], False
```
- Checks if the vector DB already exists
- If it exists → returns `([], False)` (no documents, no build needed)
- If it does **not** exist:
  1. Creates `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)` with separators `["\n\n", "\n", ". ", " ", ""]` (paragraph → line → sentence → word → char priority)
  2. Finds all `data/*.pdf` files (raises error if none)
  3. For each PDF: extracts text page-by-page via `pymupdf`
  4. Runs `smart_split` on each PDF's full text
  5. Chunks each block with the splitter
  6. Returns collected documents + `True` (build needed)

### `get_vector_store(documents, add_docs)` (lines 210-246)
- Creates `OllamaEmbeddings(model="mxbai-embed-large")` for converting text → vectors
- Opens/sets up the Chroma store at `./chroma_db`, collection `tolkien_lore`
- If building (first run):
  - Generates UUIDs for each chunk
  - Batches embeddings in groups of 50 (to avoid memory spikes), prints progress
  - Upserts (insert/update) IDs + vectors + texts + metadata into Chroma

### `get_bm25_retriever(documents, k=6)` (lines 249-256)
- Downloads NLTK punkt tokenizer data (quietly)
- Tokenizes all document contents (lowercase word split)
- Builds `BM25Okapi` index
- Returns a `BM25Retriever` wrapping the index

### `get_retriever(k=6)` (lines 259-307)
This is the **master retrieval function** that ties everything together:

1. **Load documents** (line 260): `documents, add_docs = load_documents()`
2. **Get vector store** (line 261): hands documents to Chroma if building
3. **Chroma retriever** (line 263): `vector_store.as_retriever(search_kwargs={"k": k})` — semantic search
4. **BM25 retriever** (lines 265-273):
   - If just built: use the in-memory documents
   - If DB already existed: pull all documents out of the Chroma collection and build BM25 from those
5. **Hybrid retriever** (lines 275-281): combines both with weights 0.6/0.4, returns top `k`
6. **Reranking** (lines 283-307):
   - Reads `COHERE_API_KEY` from environment
   - If key present:
     - Creates `CohereReranker`
     - Defines inner class `RerankedRetriever` which runs hybrid search, reranks with Cohere, then **filters** results keeping only those with `relevance_score >= 0.3`
     - Returns the `RerankedRetriever`
   - If no key: prints warning and returns plain hybrid retriever (no rerank)

---

## 3. requirements.txt

| Package | Purpose |
|---------|---------|
| `langchain-core` | Core LangChain abstractions (Document, BaseRetriever, prompts) |
| `langchain-community` | Community integrations (will be sunset; used for some utilities) |
| `langchain-ollama` | Ollama LLM + embeddings integration |
| `langchain-chroma` | Chroma vector store integration |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| `pypdf` | Legacy PDF reader (unused directly, kept for compatibility) |
| `PyMuPDF` | PDF text extraction (`pymupdf`) |
| `fastapi` | Web framework (installed, not yet used by the CLI app) |
| `uvicorn` | ASGI server for FastAPI (installed, not yet used) |
| `rank_bm25` | BM25 keyword scoring |
| `cohere` | Cohere reranking API |
| `nltk` | Tokenizer support for BM25 |

---

## 4. .gitignore

```
chroma_db/
__pycache__/
shire.jpg
data
```

Excludes from git:
- `chroma_db/` — vector store (rebuildable)
- `__pycache__/` — compiled Python bytecode
- `shire.jpg` — logo image
- `data` — source PDF documents

---

## Runtime Flow Summary

**First run (no chroma_db):**
1. `load_documents()` reads all 9 PDFs → smart-splits → chunks into ~800-char pieces
2. `get_vector_store()` embeds all chunks with `mxbai-embed-large` → stores in `chroma_db/`
3. BM25 index built from the same chunks
4. User asks a question → HybridRetriever → Cohere rerank → threshold filter → prompt → Qwen2.5 → answer

**Subsequent runs (chroma_db exists):**
1. `load_documents()` returns immediately (DB exists)
2. Chroma loads from disk; BM25 rebuilt from stored documents
3. Same retrieval + generation flow