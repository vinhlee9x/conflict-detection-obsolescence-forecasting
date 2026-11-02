"""Export a few projects of the Public Jira Dataset (Montgomery et al., MSR 2022).

The dump is a MongoDB database (one collection per Jira instance) holding issues
in Jira REST format. Restore it with `mongorestore`, then call `export_project`.
"""

from collections.abc import Iterable

import pandas as pd

LABEL_LINK_TYPES = ("Duplicate", "Supersede", "Supercedes", "Replace")


def issues_from_mongo(uri: str, database: str, collection: str, project_keys: str | list[str]) -> Iterable[dict]:
    from pymongo import MongoClient

    client = MongoClient(uri)
    keys = [project_keys] if isinstance(project_keys, str) else list(project_keys)
    query = {"fields.project.key": {"$in": keys}}
    projection = {
        "key": 1,
        "fields.project.key": 1,
        "fields.summary": 1,
        "fields.description": 1,
        "fields.created": 1,
        "fields.updated": 1,
        "fields.status.name": 1,
        "fields.resolution.name": 1,
        "fields.resolutiondate": 1,
        "fields.issuetype.name": 1,
        "fields.issuelinks": 1,
    }
    yield from client[database][collection].find(query, projection)


def export_issues(issues: Iterable[dict], tenant: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs, edges = [], []
    for issue in issues:
        fields = issue.get("fields", {})
        key = issue["key"]
        summary = fields.get("summary") or ""
        description = fields.get("description") or ""
        docs.append({
            "doc_id": key,
            "tenant": tenant or (fields.get("project") or {}).get("key") or key.split("-")[0],
            "root_id": key,
            "title": summary,
            "text": f"{summary}\n{description}".strip(),
            "issue_type": (fields.get("issuetype") or {}).get("name"),
            "status": (fields.get("status") or {}).get("name"),
            "resolution": (fields.get("resolution") or {}).get("name"),
            "issued_at": fields.get("created"),
            "updated_at": fields.get("updated"),
            "resolved_at": fields.get("resolutiondate"),
        })
        for link in fields.get("issuelinks") or []:
            link_type = (link.get("type") or {}).get("name", "")
            if "outwardIssue" in link:
                edges.append({"src": key, "dst": link["outwardIssue"]["key"], "type": link_type})
            elif "inwardIssue" in link:
                edges.append({"src": link["inwardIssue"]["key"], "dst": key, "type": link_type})

    docs_df = pd.DataFrame(docs)
    for column in ("issued_at", "updated_at", "resolved_at"):
        docs_df[column] = pd.to_datetime(docs_df[column], utc=True, errors="coerce").dt.tz_localize(None)

    edges_df = pd.DataFrame(edges, columns=["src", "dst", "type"]).drop_duplicates()
    known = set(docs_df["doc_id"])
    edges_df = edges_df[edges_df["src"].isin(known) & edges_df["dst"].isin(known) & (edges_df["src"] != edges_df["dst"])]
    edges_df = edges_df.assign(is_label_edge=edges_df["type"].map(is_label_link_type)).reset_index(drop=True)
    return docs_df, edges_df


def is_label_link_type(link_type: str) -> bool:
    return any(link_type.lower().startswith(t.lower()) for t in LABEL_LINK_TYPES)


def export_project(uri: str, database: str, collection: str, project_key: str | list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return export_issues(issues_from_mongo(uri, database, collection, project_key))
