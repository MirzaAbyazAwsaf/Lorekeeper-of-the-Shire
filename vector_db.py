from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import glob
import os
import re
import uuid


BOOK_NAME_MAP = {
    "the_hobbit": "The Hobbit",
}

CHAPTER_RE = re.compile(r"Chapter\s+\d+", re.IGNORECASE)


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


def smart_split(text, source, book_name):
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


def get_vector_store():
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        client_kwargs={"trust_env": False},
    )

    db_location = "./chroma_db"
    add_docs = not os.path.exists(db_location)

    documents = []
    if add_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        pdf_files = glob.glob("data/*.pdf")
        if not pdf_files:
            raise FileNotFoundError(
                "No PDF files found in data/. "
                "Place your Tolkien PDF files in the data/ folder first."
            )

        for pdf in pdf_files:
            book_name = extract_book_name(pdf)
            text = ""
            with pymupdf.open(pdf) as doc:
                for page in doc:
                    page_text = page.get_text()
                    if page_text.strip():
                        text += "\n" + page_text.strip()
            for block in smart_split(text, pdf, book_name):
                documents.extend(splitter.split_documents([block]))

        print(f"Loaded {len(documents)} chunks from {len(pdf_files)} PDF(s).")

    vector_store = Chroma(
        collection_name="tolkien_lore",
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    if add_docs:
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


def get_retriever(k=6):
    return get_vector_store().as_retriever(search_kwargs={"k": k})
