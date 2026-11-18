"""
LLM engine: the ONE and ONLY LLM in this project is Gemini (free tier).
Builds a grounded prompt from retrieved chunks + knowledge graph facts +
conversation history, and calls Gemini once to generate the answer.
If the Gemini call fails, the top retrieved chunks are printed in the terminal.
"""

from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


def build_prompt(query: str, retrieved_chunks: list, kg_facts: list, history: list) -> str:
    """Assemble everything the LLM needs into one grounded prompt."""

    context_section = "\n\n".join(f"- {chunk['text']}" for chunk in retrieved_chunks) or "No matching document context found."
    kg_section = "\n".join(f"- {fact}" for fact in kg_facts) or "No related entities found."

    history_section = ""
    if history:
        for turn in history[-3:]:  # keep last 3 turns to stay lightweight
            history_section += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    prompt = f"""You are a helpful assistant answering questions using ONLY the provided context.
If the context does not contain the answer, say so honestly instead of guessing.

--- Conversation so far ---
{history_section or "No prior conversation."}

--- Retrieved document context (from hybrid search) ---
{context_section}

--- Related entities (from knowledge graph) ---
{kg_section}

--- Current question ---
{query}

Answer clearly and concisely, grounded in the context above.
"""
    return prompt


def _retrieval_only_answer(retrieved_chunks: list, kg_facts: list) -> str:
    """Fallback text used when Gemini fails: the local retrieval results."""
    lines = ["⚠️ Gemini is unavailable right now. Here are the top chunks retrieved "
             "locally by hybrid search (BM25 + vector + RRF):\n"]
    if retrieved_chunks:
        for i, chunk in enumerate(retrieved_chunks, start=1):
            lines.append(f"[{i}] {chunk['text']}\n")
    else:
        lines.append("No matching chunks found.\n")
    if kg_facts:
        lines.append("Knowledge graph facts:")
        lines.extend(f"- {fact}" for fact in kg_facts)
    return "\n".join(lines)


def generate_answer(query: str, retrieved_chunks: list, kg_facts: list, history: list) -> str:
    """Call Gemini once. If it fails, print the top retrieved chunks in the terminal."""
    prompt = build_prompt(query, retrieved_chunks, kg_facts, history)

    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"\n[llm] Gemini failed: {e}")
        answer = _retrieval_only_answer(retrieved_chunks, kg_facts)
        print("=" * 70)
        print(f"QUERY: {query}")
        print(answer)
        print("=" * 70 + "\n")
        return answer