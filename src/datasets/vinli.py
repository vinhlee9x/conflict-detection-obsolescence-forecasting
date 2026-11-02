"""ViNLI (Huynh et al., COLING 2022) loader.

The default source is a public Hugging Face mirror of the full 4-label release;
the official copy is distributed by the UIT NLP group. Pass another base URL if
the group obtains it there.
"""

import unicodedata

import pandas as pd

MIRROR_BASE = "https://huggingface.co/datasets/presencesw/vinli_4_label/resolve/main/data"
SPLIT_FILES = {"train": "train-00000-of-00001.parquet", "dev": "dev-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
LABELS = ("entailment", "neutral", "contradiction")


def load_vinli(split: str = "test", base_url: str = MIRROR_BASE) -> pd.DataFrame:
    frame = pd.read_parquet(f"{base_url}/{SPLIT_FILES[split]}")
    return prepare(frame)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={"sentence1": "premise", "sentence2": "hypothesis", "gold_label": "label"})
    frame = frame[frame["label"].isin(LABELS)].copy()
    for column in ("premise", "hypothesis"):
        frame[column] = frame[column].map(lambda t: unicodedata.normalize("NFC", str(t)).strip())
    return frame[["premise", "hypothesis", "label"]].reset_index(drop=True)


def stratified_sample(frame: pd.DataFrame, per_label: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(n=min(per_label, len(g)), random_state=seed) for _, g in frame.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)
