import re
import time
import unicodedata

import numpy as np
import pandas as pd

from src.settings import CHUNK_MAX_CHARS, EMBEDDING_MODEL

CLAUSE_START = re.compile(r"^(?:(?:Điều|Mục)\s+\d+[.:]?|\d+\.)\s", re.MULTILINE)
DATE_PATTERN = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
LEADING_ZERO_NUMBER = re.compile(r"\b0+(\d+)\b")
WHITESPACE = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = LEADING_ZERO_NUMBER.sub(r"\1", text)
    text = DATE_PATTERN.sub(lambda m: f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}", text)
    lines = [WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def chunk(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Pack clauses into chunks of at most max_chars, never splitting a clause
    unless the clause alone is longer than max_chars."""
    starts = [m.start() for m in CLAUSE_START.finditer(text)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    units = [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]
    units = [u for u in units if u]

    chunks: list[str] = []
    current = ""
    for unit in units:
        for piece in _split_long(unit, max_chars):
            candidate = f"{current}\n{piece}" if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(unit: str, max_chars: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]
    return [unit[i:i + max_chars] for i in range(0, len(unit), max_chars)]


def build_chunks(docs: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    rows = []
    for doc_id, text in zip(docs["doc_id"], docs[text_column]):
        for index, piece in enumerate(chunk(normalize(text))):
            rows.append({"doc_id": doc_id, "chunk_index": index, "text": piece})
    return pd.DataFrame(rows)


class BgeM3Embedder:
    name = "bge-m3"

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 200
        )
        return np.asarray(vectors, dtype=np.float32)


class TfidfEmbedder:
    """Character n-gram TF-IDF reduced with SVD; a lexical baseline that needs no GPU or download."""

    name = "tfidf"

    def __init__(self, dim: int = 256, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def encode(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize as l2_normalize

        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform(texts)
        dim = min(self.dim, matrix.shape[1] - 1, matrix.shape[0] - 1)
        reduced = TruncatedSVD(n_components=dim, random_state=self.seed).fit_transform(matrix)
        return l2_normalize(reduced).astype(np.float32)


def get_embedder(backend: str):
    if backend == "bge-m3":
        return BgeM3Embedder()
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")


def embed_chunks(chunks: pd.DataFrame, embedder) -> dict:
    started = time.perf_counter()
    vectors = embedder.encode(chunks["text"].tolist())
    elapsed = time.perf_counter() - started
    return {
        "vectors": vectors.astype(np.float16),
        "doc_ids": chunks["doc_id"].to_numpy(dtype=str),
        "chunk_index": chunks["chunk_index"].to_numpy(dtype=np.int32),
        "backend": np.array(embedder.name),
        "seconds": np.array(elapsed),
    }


def ocr_pdf(path: str, lang: str = "vie") -> str:
    """OCR a scanned PDF with Tesseract; needs pytesseract, pdf2image and the vie language pack."""
    import pytesseract
    from pdf2image import convert_from_path

    return "\n".join(pytesseract.image_to_string(page, lang=lang) for page in convert_from_path(path))
