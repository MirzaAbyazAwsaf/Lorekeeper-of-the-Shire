from langchain_ollama.llms import OllamaLLM
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
    model = OllamaLLM(model="qwen2.5:7b-instruct-q4_K_M", client_kwargs={"trust_env": False})

    template = """
You are a world-renowned Tolkien scholar and lore-master with encyclopedic knowledge
of J.R.R. Tolkien's works. Currently your expertise covers The Hobbit in full detail,
including every character, event, location, theme, theory behind it.

IMPORTANT:
- If {passages} is empty or contains no relevant information, this question is NOT about
  Tolkien lore. Respond naturally - greet the user, chat, or say the question is outside
  your knowledge of Middle-earth. Do NOT make up an answer about Middle-earth.
- If {passages} contains relevant information, answer ONLY using those passages.
- If the passages do not contain enough information to fully answer, say so honestly.
- Never fabricate details that are not present in the provided context.

When answering lore questions:
- Be warm and hospitable to the user.
- Cite the book and chapter when relevant
- Provide direct quotes from the text when possible
- Explain connections between events, characters, and themes
- Mention any well-known theories or interpretations when relevant
- Give thorough, detailed answers — this user wants deep lore knowledge

Relevant passages:
{passages}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("=" * 60)
    print(" Lorekeeper of the Shire - The Hobbit")
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
        results = chain.invoke({"passages": format_passages(docs), "question": question})
        print(f"\n{results}")


if __name__ == "__main__":
    main()




