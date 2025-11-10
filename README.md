# HEB Product Retrieval and Reranking System

This project implements a **hybrid retrieval and reranking pipeline** for matching queries to product descriptions.  
It combines **OpenAI embeddings**, **FAISS (or NumPy fallback)** for similarity search, and a **cross-encoder model** for reranking.  
The system can be used for query–product retrieval tasks such as e-commerce search, information retrieval, and synthetic query benchmarking.

---

## Features

- Text preprocessing and normalization (stopword removal, lowercase, alphanumeric filtering)
- Embedding of product metadata fields (`title`, `brand`, `category`, `description`) using OpenAI embeddings
- Construction of an inner-product (cosine similarity) index using FAISS or a NumPy fallback
- Query embedding and top-K retrieval
- Cross-encoder reranking using a transformer model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Batch processing of all queries to generate ranked product predictions
- JSON output compatible with evaluation pipelines (e.g., `submission.json` format)

---

## Installation

### Requirements

- Python 3.9+
- Dependencies:
  ```bash
  pip install -r requirements.txt



HOW TO RUN:
/usr/local/bin/python3 PATH/semanticSearch.py --products [PATH_TO_PRODUCTS_FILE] --queries [PATH_TO_QUERIES_FILE]
