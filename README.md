# 🎯 Multimodal Stateful Hybrid RAG System

A Multimodal Stateful Hybrid Retrieval-Augmented Generation (RAG) System that combines hybrid search, knowledge graphs, cross-lingual information retrieval, and conversational memory to provide context-aware answers from uploaded documents.

#Overview

This project implements a Streamlit-based application that allows users to upload documents in multiple formats and ask questions about their content.

It combines keyword-based retrieval and semantic vector search to improve information retrieval. Knowledge Graph integration provides additional entity-based context, while LangGraph maintains conversational state across multiple interactions.

The system supports queries and responses in English, Tamil, and Hindi. Google Gemini is used as the language model for response generation.

#Key Features

Multimodal Document Ingestion: Supports PDF, DOCX, images, and audio files.

Text Extraction: Extracts text from documents using format-specific processing, OCR, and speech-to-text.

Hybrid Search: Combines BM25 keyword retrieval and ChromaDB semantic vector retrieval.

Reciprocal Rank Fusion (RRF): Combines rankings from multiple retrieval methods.

Cross-Encoder Re-ranking: Re-ranks retrieved passages to improve relevance.

Knowledge Graph: Uses spaCy and NetworkX for entity extraction and relationship-based context.

Cross-Lingual Retrieval: Supports English, Tamil, and Hindi queries.

Multilingual Response Generation: Generates responses in the user's supported language.

Stateful Conversation: Uses LangGraph to maintain context across multiple conversation turns.

Interactive Interface: Provides a Streamlit web interface for document upload and question answering.

#System Workflow

Upload PDF, DOCX, image, or audio files.

Extract text using document readers, OCR, or speech-to-text.

Clean and normalize extracted text.

Split text into overlapping chunks.

Generate embeddings and store them in ChromaDB.

Build a BM25 keyword retrieval index.

Retrieve relevant chunks using hybrid search.

Combine rankings using Reciprocal Rank Fusion.

Re-rank retrieved results using a cross-encoder.

Retrieve related entity information from the Knowledge Graph.

Combine retrieved context with conversation history.

Generate a context-aware response using Google Gemini.
