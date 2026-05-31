"""
ingest.py — Parte 1 y 5: Ingesta, limpieza, chunking e indexación.

Estrategia de embeddings:
  - Intenta usar sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
    para búsqueda semántica de alta calidad (requiere descarga del modelo ~120 MB).
  - Si el modelo no está disponible, usa TF-IDF como fallback (funciona offline).
El archivo index_meta.pkl guarda chunks + vectores + modo usado.
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path

import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DOCS_DIR   = Path(__file__).parent.parent / "docs"
META_PATH  = Path(__file__).parent / "index_meta.pkl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 400 #corta en pedazos de 400 caracteres
CHUNK_OVERLAP = 80 #80 caracteres de solapamiento para no perder contexto en bordes


# ── Lectores ──────────────────────────────────────────────────────────────────

def read_txt(path):  return path.read_text(encoding="utf-8", errors="ignore")
def read_md(path):   return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: pages.append(t)
    return "\n\n".join(pages)

def read_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("contenido", [data])
    parts = []
    for item in items:
        if isinstance(item, dict):
            rows = []
            for k, v in item.items():
                if isinstance(v, list): v = "; ".join(str(x) for x in v)
                rows.append(f"{k}: {v}")
            parts.append("\n".join(rows))
        else:
            parts.append(str(item))
    return "\n\n".join(parts)

READERS = {".pdf": read_pdf, ".txt": read_txt, ".md": read_md, ".json": read_json}


# ── Limpieza ──────────────────────────────────────────────────────────────────

def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_chunks(text, source, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append({"text": current, "source": source})
            if len(para) > size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= size:
                        buf = (buf + " " + sent).strip()
                    else:
                        if buf: chunks.append({"text": buf, "source": source})
                        buf = sent
                current = buf
            else:
                tail = current[-overlap:] if overlap < len(current) else current
                current = (tail + "\n\n" + para).strip()
    if current:
        chunks.append({"text": current, "source": source})
    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────

def build_embeddings_sentence_transformers(texts):
    from sentence_transformers import SentenceTransformer
    log.info("Usando sentence-transformers para embeddings semánticos...")
    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    # Normalizar para similitud coseno
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms, "sentence-transformers"


def build_embeddings_tfidf(texts):
    log.info("Usando TF-IDF como método de indexación (fallback offline)...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts).toarray().astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms, "tfidf", vectorizer


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    log.info(f"Indexando {len(texts)} fragmentos...")

    try:
        import faiss
        vecs, mode = build_embeddings_sentence_transformers(texts)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        meta = {
            "mode": mode,
            "chunks": chunks,
            "faiss_index": index,
        }
        log.info(f"Índice FAISS creado (dim={dim}, {index.ntotal} vectores).")
    except Exception as e:
        log.warning(f"sentence-transformers/FAISS no disponible ({e}). Usando TF-IDF.")
        vecs, mode, vectorizer = build_embeddings_tfidf(texts)
        meta = {
            "mode": mode,
            "chunks": chunks,
            "vectors": vecs,
            "vectorizer": vectorizer,
        }
        log.info(f"Índice TF-IDF creado ({vecs.shape[0]} fragmentos, dim={vecs.shape[1]}).")

    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    log.info(f"Índice guardado en: {META_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_documents(docs_dir):
    all_chunks = []
    for file in sorted(docs_dir.iterdir()):
        ext = file.suffix.lower()
        if ext not in READERS:
            log.warning(f"Formato no soportado, omitiendo: {file.name}")
            continue
        log.info(f"Leyendo: {file.name}")
        try:
            raw   = READERS[ext](file)
            clean = clean_text(raw)
            cks   = split_into_chunks(clean, source=file.name)
            log.info(f"  → {len(cks)} fragmentos generados")
            all_chunks.extend(cks)
        except Exception as e:
            log.error(f"Error procesando {file.name}: {e}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_documents(DOCS_DIR)
    if not chunks:
        log.error("No se encontraron documentos.")
        raise SystemExit(1)
    build_index(chunks)
    log.info("Ingesta completada.")
