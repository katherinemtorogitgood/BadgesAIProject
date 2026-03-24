# BadgesAIProject

A local retrieval-augmented generation (RAG) web app for IBM Z and mainframe badge study help. It answers questions from your own uploaded text files instead of relying only on the base model.

## What it does

This project lets you:

- ingest `.txt` study materials into a ChromaDB vector database
- ask questions through a FastAPI backend and simple HTML frontend
- get grounded answers based on the files you added
- support badge topics like:
  - Introduction to IBM z/OS
  - Commands and Panels on IBM Z
  - Introduction to System Programming on IBM Z
  - Introduction to z/OS UNIX System Services
  - Modernizing Applications with IBM CICS
  - TCP/IP Protocol Overview

## Tech stack

- Python
- FastAPI
- Ollama
- ChromaDB
- HTML / CSS / JavaScript

## How it works

1. Put source `.txt` files into `input_texts/`
2. Run the ingestion script to chunk, embed, and store the text in ChromaDB
3. Start the FastAPI backend
4. Open the frontend and ask questions
5. The app retrieves relevant chunks and sends them to the LLM for a grounded answer

## Project structure

```text
BadgesAIProject/
├── app.py
├── ingest.py
├── index.html
├── requirements.txt
├── input_texts/
├── chroma_db/          # generated locally after ingestion
└── README.md
