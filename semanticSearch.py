
from __future__ import annotations

import argparse
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# .env support (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# First lets embed all products and put them in a vector storage.
from openai import OpenAI

_OPENAI_CLIENT = None

def get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment. "
                "Set it (export OPENAI_API_KEY=...) or put it in a .env file."
            )
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT

def embed_texts_openai(texts: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
    client = get_openai_client()
    processed = [(t if str(t).strip() != "" else " ") for t in texts]
    resp = client.embeddings.create(model=model, input=processed)
    V = np.array([d.embedding for d in resp.data], dtype="float32")
    # L2 normalize → cosine == inner product
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


# Now we set up the index we will use for similarity (FAISS) which uses cosine similarity.
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

def build_ip_index(emb: np.ndarray):
    """
    Inner-product index. Uses FAISS if available, or a NumPy fallback with
    the same .search(Q, k) API. Assumes embeddings are L2-normalized.
    """
    if FAISS_AVAILABLE:
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(emb.astype("float32"))
        return index

    class NumpyIPIndex:
        def __init__(self, X):
            self.X = X.astype("float32")
        def add(self, _):
            pass
        def search(self, Q, k):
            S = Q @ self.X.T
            I = np.argpartition(-S, k-1, axis=1)[:, :k]
            S_topk = np.take_along_axis(S, I, axis=1)
            order = np.argsort(-S_topk, axis=1)
            I_sorted = np.take_along_axis(I, order, axis=1)
            D_sorted = np.take_along_axis(S_topk, order, axis=1)
            return D_sorted.astype("float32"), I_sorted.astype("int64")
    return NumpyIPIndex(emb)

def save_faiss_index(index, path: Path):
    if FAISS_AVAILABLE and isinstance(index, faiss.Index):
        faiss.write_index(index, str(path))

def load_faiss_index(path: Path):
    if FAISS_AVAILABLE and path.exists():
        return faiss.read_index(str(path))
    return None


# PREPROCESSING TEXT. This is particularly important for the description since it can have a lot of fluff and reduce the performance.
STOPWORDS = set(["a","an","the","and","is","in","it","to","for","with","on","this","that"])

def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9%\. ]+", " ", s)
    s = " ".join([w for w in s.split() if w not in STOPWORDS])
    return s.strip()


# Now let's load in the data. First, we load in products.json to build vector database. 
# Then, we load in either the train or test data based on what we want to create predictions for. 
def resolve_existing_file(preferred: str, keywords=("products",), exts=(".json", ".jsonl")) -> Path:
    p = Path(preferred)
    if p.exists():
        return p
    roots = [Path("."), Path.cwd()]
    for root in roots:
        for ext in exts:
            for cand in root.rglob(f"*{ext}"):
                if all(k in cand.name.lower() for k in keywords):
                    return cand
    raise FileNotFoundError(f"Could not find file like '{preferred}' with keywords={keywords} and exts={exts}")

# Load the products in
def load_products_any(products_path_hint: str) -> List[Dict[str, Any]]:
    fp = resolve_existing_file(products_path_hint, keywords=("products",), exts=(".json",".jsonl"))
    # JSON array or dict
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "products" in data:
            return data["products"]
    except json.JSONDecodeError:
        pass
    # JSON Lines fallback
    items = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

# Load in the queries. 
def load_queries_any(path_hint: str) -> List[Dict[str, str]]:
    fp = resolve_existing_file(path_hint, keywords=("queries","synth"), exts=(".json",".jsonl"))
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict) and "queries" in data:
            raw = data["queries"]
        else:
            raise ValueError("Unsupported JSON structure for queries")
    except json.JSONDecodeError:
        raw = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    out = []
    for j in raw:
        qid = j.get("query_id") or j.get("qid") or j.get("id")
        qtx = j.get("query") or j.get("text")
        if qid and qtx:
            out.append({"query_id": str(qid), "query": str(qtx)})
    return out


# Now we actually embed products.json and store in vector database.
def batch_embed(texts: Iterable[str], embed_fn, batch_size: int = 256) -> np.ndarray:
    texts = list(texts)
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        out.append(embed_fn(texts[i:i+batch_size]))
    return np.vstack(out) if out else np.zeros((0, 384), dtype="float32")

def build_vector_store(
    products_json_path: str,
    out_dir: str = "vector_store"
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load products
    products = load_products_any(products_json_path)
    if not products:
        raise ValueError("No products loaded. Check your products file.")

    # 2) Canonicalize & normalize
    rows = []
    for r in products:
        pid = str(r.get("product_id") or r.get("id") or r.get("sku") or "")
        title = r.get("title") or r.get("name") or ""
        brand = r.get("brand") or ""
        category = r.get("category_path") or r.get("category") or ""
        desc = r.get("description") or ""
        rows.append({
            "product_id": pid,
            "title": title,
            "brand": brand,
            "category_path": category,
            "description": desc,
            "norm_title": norm_text(title),
            "norm_brand": norm_text(brand),
            "norm_category": norm_text(category),
            "norm_description": norm_text(desc),
        })
    df = pd.DataFrame(rows)
    (out / "products.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "products.parquet", index=False)

    # 3) Embeddings per field (OpenAI)
    print("Embedding title...")
    emb_title = batch_embed(df["norm_title"].fillna("").astype(str).tolist(), embed_texts_openai)
    print("Embedding brand...")
    emb_brand = batch_embed(df["norm_brand"].fillna("").astype(str).tolist(), embed_texts_openai)
    print("Embedding category...")
    emb_cat   = batch_embed(df["norm_category"].fillna("").astype(str).tolist(), embed_texts_openai)
    print("Embedding description...")
    emb_desc  = batch_embed(df["norm_description"].fillna("").astype(str).tolist(), embed_texts_openai)

    # 4) Concatenate embeddings → one big matrix
    emb_combined = np.concatenate([emb_title, emb_brand, emb_cat, emb_desc], axis=1)
    np.save(out / "emb_combined.npy", emb_combined)

    # 5) Build & save FAISS index (if available)
    print("Building combined index (FAISS available? ->", FAISS_AVAILABLE, ")")
    idx = build_ip_index(emb_combined)
    if FAISS_AVAILABLE:
        save_faiss_index(idx, out / "index_combined.faiss")

    # 6) Row → product_id map
    pid_map = df["product_id"].astype(str).tolist()
    pd.Series(pid_map).to_json(out / "rowid_to_pid.json", orient="values")

    print("✅ Vector store written to:", out.resolve())
    return out

# ---------------------
# Load vector store
# ---------------------
@dataclass
class VectorStore:
    root: Path
    E_combined: np.ndarray
    IDX_combined: Any
    rowid_to_pid: List[str]
    meta: pd.DataFrame

def load_vector_store(vstore_dir: str = "vector_store") -> VectorStore:
    root = Path(vstore_dir)
    E_combined = np.load(root / "emb_combined.npy")
    # Load FAISS if present, else build in-memory index
    idx_path = root / "index_combined.faiss"
    if FAISS_AVAILABLE and idx_path.exists():
        IDX_combined = load_faiss_index(idx_path)
        if IDX_combined is None:
            IDX_combined = build_ip_index(E_combined)
    else:
        IDX_combined = build_ip_index(E_combined)

    rowid_to_pid = json.loads((root / "rowid_to_pid.json").read_text())
    meta = pd.read_parquet(root / "products.parquet")
    return VectorStore(root, E_combined, IDX_combined, rowid_to_pid, meta)

# -------------------------
# Cross-encoder reranker
# -------------------------
from sentence_transformers import CrossEncoder

def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    return CrossEncoder(model_name)

def product_text_from_row(row: pd.Series) -> str:
    # Short, structured layout helps CEs
    return (f"[TITLE] {row['title']} [SEP] [BRAND] {row['brand']} [SEP] "
            f"[CAT] {row['category_path']} [SEP] "
            f"[DESC] {(row['description'] or '')[:300]}")


# Now we search through and find the highest 50 in similarity in comparison to the query.
def search(query_text: str,
           vs: VectorStore,
           cross_encoder_model: CrossEncoder,
           k_return: int = 10,
           k_rerank: int = 50,
           w_title: float = 0.80,
           w_brand: float = 0.05,
           w_cat: float = 0.10,
           w_desc: float = 0.05) -> List[Dict[str, Any]]:
    """
    1) normalize + embed query once (OpenAI)
    2) build weighted combined query vector
    3) retrieve top-k_rerank from the combined index
    4) cross-encode top candidates and sort
    5) return top-k_return
    """
    qn = norm_text(query_text)

    # Embed query once (reuse for all four fields)
    qv = embed_texts_openai([qn])  # (1, D)
    qv_combined = np.concatenate([
        qv * w_title,
        qv * w_brand,
        qv * w_cat,
        qv * w_desc
    ], axis=1)

    D, I = vs.IDX_combined.search(qv_combined, k_rerank)
    cand_ids = I[0].tolist()

    sentence_pairs = []
    items = []
    for i in cand_ids:
        row = vs.meta.iloc[i]
        pid = vs.rowid_to_pid[i]
        sentence_pairs.append([query_text, product_text_from_row(row)])
        items.append({
            "product_id": pid,
            "name": row["title"],
            "brand": row["brand"],
            "category": row["category_path"],
        })

    if not sentence_pairs:
        return []

    ce_scores = cross_encoder_model.predict(sentence_pairs)
    for it, s in zip(items, ce_scores):
        it["score"] = float(s)

    items.sort(key=lambda r: r["score"], reverse=True)
    return items[:k_return]


# Now lets rerank the top 50 in similarity with the cross encoder. This will reduce the amount of false positives we come across.
def rank_all_queries_topk_reranked(queries: List[Dict[str, str]],
                                   vs: VectorStore,
                                   ce: CrossEncoder,
                                   k_return: int = 30,
                                   k_rerank: int = 50) -> List[Dict[str, Any]]:
    out = []
    print(f"Re-ranking {len(queries)} queries (k_rerank={k_rerank} → k_return={k_return})...")
    for i, q in enumerate(tqdm(queries, desc="Queries")):
        qid, qtext = q["query_id"], q["query"]
        results = search(qtext, vs, ce, k_return=k_return, k_rerank=k_rerank)
        # deterministic tie-break
        results = sorted(results, key=lambda r: (-r["score"], str(r["product_id"])))
        for rank, r in enumerate(results, start=1):
            out.append({
                "query_id": qid,
                "rank": rank,
                "product_id": str(r["product_id"]),
            })
    return out

# Main function that will actually run everything. 
def main():
    ap = argparse.ArgumentParser(description="HEB XOXO: build, retrieve, rerank")
    ap.add_argument("--products", required=True, help="Path to products.json or .jsonl")
    ap.add_argument("--queries", required=True, help="Path to queries_synth_*.json")
    ap.add_argument("--out", default="submission_reranked.json", help="Output JSON path (submission format)")
    ap.add_argument("--vstore", default="vector_store", help="Vector store directory")
    ap.add_argument("--k_rerank", type=int, default=50, help="Top-K from retrieval to feed cross-encoder")
    ap.add_argument("--k_return", type=int, default=30, help="Final top-K to output")
    ap.add_argument("--w_title", type=float, default=0.80)
    ap.add_argument("--w_brand", type=float, default=0.05)
    ap.add_argument("--w_cat", type=float, default=0.10)
    ap.add_argument("--w_desc", type=float, default=0.05)
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store")
    args = ap.parse_args()

    vstore_root = Path(args.vstore)
    need_build = args.rebuild or not (vstore_root / "emb_combined.npy").exists()

    if need_build:
        print("Building vector store...")
        build_vector_store(products_json_path=args.products, out_dir=args.vstore)
    else:
        print("Using existing vector store at:", vstore_root.resolve())

    print("Loading vector store and cross-encoder...")
    vs = load_vector_store(args.vstore)
    ce = load_cross_encoder()

    print("Loading queries...")
    queries = load_queries_any(args.queries)
    if not queries:
        raise RuntimeError("No queries loaded. Check --queries path.")

    print(f"Running batch ranking for {len(queries)} queries...")
    flat = rank_all_queries_topk_reranked(
        queries,
        vs,
        ce,
        k_return=args.k_return,
        k_rerank=args.k_rerank
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2))
    print("Wrote submission to:", out_path.resolve())

if __name__ == "__main__":
    main()
