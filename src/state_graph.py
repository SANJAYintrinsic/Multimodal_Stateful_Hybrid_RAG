"""
Stateful conversation flow using LangGraph.
Nodes: detect language -> translate query -> hybrid+KG retrieve
       -> generate answer (Gemini) -> translate back -> update memory.
This is what makes the system "stateful" (remembers prior turns)
rather than answering each question in isolation.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.translator import detect_language, translate_to_english, respond_in_original_language
from src.llm_engine import generate_answer


class RAGState(TypedDict):
    query: str                 # original user query (any language)
    query_en: str               # query translated to English for retrieval
    language: str                # "en" or "ta"
    retrieved_chunks: list
    kg_facts: list
    answer_en: str              # answer in English (from the LLM)
    final_answer: str            # answer translated back to user's language
    history: list                 # list of {"user": ..., "assistant": ...}


def build_rag_graph(search_engine, kg_builder):
    """
    search_engine: an instance of HybridSearchEngine
    kg_builder: an instance of KnowledgeGraphBuilder
    These are passed in so the graph reuses the SAME indexes/graph
    across every turn, instead of rebuilding them each time.
    """

    def detect_language_node(state: RAGState) -> RAGState:
        state["language"] = detect_language(state["query"])
        return state

    def translate_query_node(state: RAGState) -> RAGState:
        state["query_en"] = translate_to_english(state["query"])
        return state

    def retrieve_node(state: RAGState) -> RAGState:
        state["retrieved_chunks"] = search_engine.hybrid_search(state["query_en"])
        state["kg_facts"] = kg_builder.related_entities(state["query_en"])
        return state

    def generate_node(state: RAGState) -> RAGState:
        state["answer_en"] = generate_answer(
            query=state["query_en"],
            retrieved_chunks=state["retrieved_chunks"],
            kg_facts=state["kg_facts"],
            history=state["history"],
        )
        return state

    def translate_back_node(state: RAGState) -> RAGState:
        state["final_answer"] = respond_in_original_language(state["answer_en"], state["query"])
        return state

    def update_memory_node(state: RAGState) -> RAGState:
        state["history"].append({"user": state["query"], "assistant": state["final_answer"]})
        return state

    graph = StateGraph(RAGState)

    graph.add_node("detect_language", detect_language_node)
    graph.add_node("translate_query", translate_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("translate_back", translate_back_node)
    graph.add_node("update_memory", update_memory_node)

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "translate_query")
    graph.add_edge("translate_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "translate_back")
    graph.add_edge("translate_back", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()