from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import pymupdf
import cohere
import glob
import os
import re
import uuid
import nltk


BOOK_NAME_MAP = {
    "the_hobbit": "The Hobbit",
}

CHAPTER_RE = re.compile(r"Chapter\s+\d+", re.IGNORECASE)
MD_HEADER_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


class BM25Retriever(BaseRetriever):
    bm25: BM25Okapi
    documents: list
    k: int = 6

    def _get_relevant_documents(self, query: str) -> list:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.k
        ]
        return [self.documents[i] for i in top_k_idx]


class HybridRetriever(BaseRetriever):
    chroma_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    chroma_weight: float = 0.6
    bm25_weight: float = 0.4
    k: int = 6

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
        return [Document(page_content=c, metadata=m) for c, m, s in scored[: self.k]]


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0", top_n: int = 6):
        self.client = cohere.Client(api_key)
        self.model = model
        self.top_n = top_n

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


def extract_book_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0].lower()
    base = re.sub(r"[\s_\-]+", "_", base)
    if base in BOOK_NAME_MAP:
        return BOOK_NAME_MAP[base]
    return os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")


def split_by_chapters(text, source, book_name):
    chapters = re.split(r"(?=Chapter\s+\d+)", text, flags=re.IGNORECASE)
    documents = []
    for block in chapters:
        block = block.strip()
        if not block:
            continue
        chapter_match = re.match(
            r"(Chapter\s+\d+)[:\.\s–—-]*(.*)", block, re.IGNORECASE
        )
        if chapter_match:
            chapter_num = chapter_match.group(1).strip()
            chapter_title = chapter_match.group(2).strip()
            chapter_label = (
                f"{chapter_num}: {chapter_title}" if chapter_title else chapter_num
            )
        else:
            chapter_label = "Preface / Introduction"
        documents.append(
            Document(
                page_content=block,
                metadata={
                    "source": source,
                    "book": book_name,
                    "chapter": chapter_label,
                },
            )
        )
    return documents


def split_by_sections(text, source, book_name, section_marker="==="):
    sections = text.split(section_marker)
    documents = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        first_line = section.split("\n", 1)[0].strip()
        documents.append(
            Document(
                page_content=section,
                metadata={
                    "source": source,
                    "book": book_name,
                    "section": first_line[:100] if first_line else f"Section {i+1}",
                },
            )
        )
    return documents


def split_by_markdown_headers(text, source, book_name):
    sections = re.split(r"(?=^#{1,6}\s+)", text, flags=re.MULTILINE)
    documents = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        header_match = re.match(r"^#{1,6}\s+(.+)$", section, re.MULTILINE)
        if header_match:
            heading = header_match.group(1).strip()
            section = re.sub(r"^#{1,6}\s+", "", section, count=1, flags=re.MULTILINE).strip()
        else:
            heading = "Introduction"
        documents.append(
            Document(
                page_content=section,
                metadata={
                    "source": source,
                    "book": book_name,
                    "section": heading,
                },
            )
        )
    return documents


def smart_split(text, source, book_name):
    if MD_HEADER_RE.search(text):
        return split_by_markdown_headers(text, source, book_name)
    if book_name == "The Hobbit" and re.search(CHAPTER_RE, text):
        return split_by_chapters(text, source, book_name)
    if "===" in text:
        return split_by_sections(text, source, book_name)
    return [
        Document(
            page_content=text.strip(),
            metadata={"source": source, "book": book_name},
        )
    ]


def load_documents():
    db_location = "./chroma_db"
    add_docs = not os.path.exists(db_location)
    if not add_docs:
        return [], False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pdf_files = glob.glob("data/*.pdf")
    md_files = glob.glob("data/*.md")
    doc_files = pdf_files + md_files
    if not doc_files:
        raise FileNotFoundError(
            "No PDF or Markdown files found in data/. "
            "Place your Tolkien PDF or Markdown files in the data/ folder first."
        )

    documents = []
    for path in doc_files:
        book_name = extract_book_name(path)
        text = ""
        if path.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            with pymupdf.open(path) as doc:
                for page in doc:
                    page_text = page.get_text()
                    if page_text.strip():
                        text += "\n" + page_text.strip()
        for block in smart_split(text, path, book_name):
            documents.extend(splitter.split_documents([block]))

    print(f"Loaded {len(documents)} chunks from {len(doc_files)} source(s).")
    return documents, True


def get_vector_store(documents=None, add_docs=False):
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        client_kwargs={"trust_env": False},
    )

    db_location = "./chroma_db"

    vector_store = Chroma(
        collection_name="tolkien_lore",
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    if add_docs and documents:
        chunk_texts = [d.page_content for d in documents]
        chunk_metas = [d.metadata for d in documents]
        ids = [str(uuid.uuid4()) for _ in chunk_texts]

        print("Embedding chunks (this may take a while)...", flush=True)
        vectors = []
        batch_size = 50
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            vectors.extend(embeddings.embed_documents(batch))
            done = min(i + batch_size, len(chunk_texts))
            print(f"  embedded {done}/{len(chunk_texts)} chunks", flush=True)

        vector_store._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=chunk_texts,
            metadatas=chunk_metas,
        )
        print(f"Vector store built and persisted to {db_location}.")

    return vector_store


def get_bm25_retriever(documents, k=6):
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    tokenized_docs = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return BM25Retriever(bm25=bm25, documents=documents, k=k)


def get_retriever(k=6):
    documents, add_docs = load_documents()
    vector_store = get_vector_store(documents, add_docs)

    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    if add_docs and documents:
        bm25_retriever = get_bm25_retriever(documents, k=k)
    else:
        all_docs = vector_store._collection.get(include=["documents", "metadatas"])
        docs_list = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        bm25_retriever = get_bm25_retriever(docs_list, k=k)

    hybrid_retriever = HybridRetriever(
        chroma_retriever=chroma_retriever,
        bm25_retriever=bm25_retriever,
        chroma_weight=0.6,
        bm25_weight=0.4,
        k=k,
    )

    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if cohere_key:
        reranker = CohereReranker(api_key=cohere_key, top_n=k)

        class RerankedRetriever(BaseRetriever):
            hybrid: HybridRetriever
            reranker: CohereReranker
            k: int = 6
            relevance_threshold: float = 0.3

            def _get_relevant_documents(self, query: str) -> list:
                docs = self.hybrid.invoke(query)
                reranked = self.reranker.rerank(query, docs)
                relevant = [
                    d for d in reranked
                    if d.metadata.get("relevance_score", 0) >= self.relevance_threshold
                ]
                return relevant

        return RerankedRetriever(
            hybrid=hybrid_retriever, reranker=reranker, k=k, relevance_threshold=0.3
        )
    else:
        print("Warning: COHERE_API_KEY not set. Skipping reranking.")
        return hybrid_retriever
