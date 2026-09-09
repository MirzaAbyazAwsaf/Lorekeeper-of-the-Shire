from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector_db import get_retriever


def format_passages(passages):
    parts = []
    for p in passages:
        book = p.metadata.get("book", "Unknown")
        chapter = p.metadata.get("chapter", p.metadata.get("section", ""))
        header = f"[{book}-{chapter}]" if chapter else f"[{book}]"
        parts.append(f"{header}\n{p.page_content}")
    return "\n\n---\n\n".join(parts)


def main():
    retriever = get_retriever()
    model = ChatOllama(model="llama3.2", client_kwargs={"trust_env": False})

    template = """You are a Tolkien genealogy expert. Output family trees ONLY.

RULES:
1. Father in the CENTER TOP
2. ALL wives connect DIRECTLY to the father (no wife-to-wife connections)
3. Children below each wife with pipes: |
4. Use horizontal dashes for spouse connections (--------)
5. Exact spacing - monospace format

2 WIVES FORMAT:
Wife1--------Father--------Wife2
       |                         |
     Child1                   Child2

3 WIVES FORMAT (every wife has its own line directly to Father):
Wife1--------Father--------Wife2
       |         |
                    |
            Wife3      |
            Child3             |
       |                   Child2
     Child1
       |
       |
     Wife3
       |
     Child3

NO explanations. ONLY output the tree.

Passages:
{passages}

Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("=" * 60)
    print(" Tolkien Family Tree Generator")
    print("  Type 'q' to quit")
    print("=" * 60)

    while True:
        question = input("\n> ")
        if question.strip().lower() == "q":
            print("Mellon nath, farewell!")
            break
        if not question.strip():
            continue

        docs = retriever.invoke(question)
        print("\n", end="", flush=True)
        for chunk in chain.stream({"passages": format_passages(docs), "question": question}):
            print(chunk.content, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
