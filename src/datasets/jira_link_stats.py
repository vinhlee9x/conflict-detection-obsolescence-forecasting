import pandas as pd


def link_type_counts(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.groupby(["type", "is_label_edge"]).size().rename("n").reset_index().sort_values("n", ascending=False)


def resolution_counts(docs: pd.DataFrame) -> pd.DataFrame:
    return docs["resolution"].fillna("(none)").value_counts().rename_axis("resolution").reset_index(name="n")


def project_summary(docs: pd.DataFrame, edges: pd.DataFrame) -> dict:
    return {
        "issues": int(len(docs)),
        "links": int(len(edges)),
        "label_links": int(edges["is_label_edge"].sum()),
        "first_created": docs["issued_at"].min(),
        "last_created": docs["issued_at"].max(),
    }
