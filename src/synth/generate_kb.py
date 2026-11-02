"""Generate the synthetic Vietnamese knowledge base with exact labels.

Generative story, per tenant:
- Each root document is a (topic, subject) pair rendered from clause templates.
- A root may get one revision in the same series: a conflicting one (a clause
  value changes, the root becomes obsolete) or a paraphrased one (wording
  changes, values stay). Both revisions cite the root, so graph edges say
  "related", never "conflicting".
- A hidden conflict restates part of a root from another department with one
  changed value and no link to it. It often appears shortly before a
  conflicting revision, which is the precursor signal Layer 3 can learn.
- A lexical-overlap negative borrows a root's title but covers another topic.
"""

from dataclasses import dataclass, field, asdict
import json
import re

import numpy as np
import pandas as pd

from src.synth.templates import TENANTS, Tenant, Topic

KB_START = pd.Timestamp("2022-01-01")
ROOT_ISSUE_END = pd.Timestamp("2023-06-30")
KB_END = pd.Timestamp("2024-12-31")

P_CONFLICT_REVISION = 0.45
P_PARAPHRASE_REVISION = 0.40
P_HIDDEN_BEFORE_REVISION = 0.6
P_HIDDEN_OTHERWISE = 0.15
P_HIDDEN_CONFIRMED = 0.5
P_LEXICAL_NEGATIVE = 0.3
NEGATIVES_PER_POSITIVE = 3
RANDOM_NEGATIVE_OVERSAMPLE = 1.5
SAME_TEMPLATE_NEGATIVE_SHARE = 0.3
DEPARTMENT_PEERS = 3
REVISION_REWORD_PROBABILITY = 0.6
P_REVISION_ADDS_CLAUSE = 0.5
P_REVISION_DROPS_CLAUSE = 0.3

CODE_PATTERN = re.compile(r"\b(?:SOP|POL)-\d{3}\b")


@dataclass
class ClauseState:
    key: str
    phrasing: int
    values: dict


@dataclass
class Doc:
    doc_id: str
    code: str
    version: int
    tenant: str
    root_id: str
    series_id: str
    topic: str
    subject: str
    department: str
    title: str
    issued_at: pd.Timestamp
    variant_type: str
    clauses: list
    boilerplate: list
    cites: list = field(default_factory=list)
    revision_of: str | None = None
    superseded_at: pd.Timestamp | None = None
    text: str = ""


def generate_kb(n_roots_per_tenant: int = 200, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    docs: list[Doc] = []
    pairs: list[dict] = []
    events: list[dict] = []
    for tenant in TENANTS.values():
        tenant_docs, tenant_pairs, tenant_events = _generate_tenant(tenant, n_roots_per_tenant, rng)
        docs.extend(tenant_docs)
        pairs.extend(tenant_pairs)
        events.extend(tenant_events)

    docs_df = _docs_frame(docs)
    edges_df = build_edges(docs_df, rng)
    pairs_df = pd.DataFrame(pairs)
    events_df = pd.DataFrame(events, columns=["doc_a", "doc_b", "tenant", "confirmed_at", "source"])
    return {"docs": docs_df, "pairs": pairs_df, "edges": edges_df, "conflict_events": events_df}


def _generate_tenant(tenant: Tenant, n_roots: int, rng: np.random.Generator):
    combos = [(t, s) for t in tenant.topics for s in tenant.subjects]
    order = rng.permutation(len(combos))[:n_roots]
    counter = _CodeCounter(tenant.code_prefix)

    roots = [_make_root(tenant, *combos[i], counter.next(), rng) for i in order]
    _add_root_citations(roots, rng)

    docs = list(roots)
    pairs: list[dict] = []
    events: list[dict] = []
    for root in roots:
        topic = _topic(tenant, root.topic)
        revision_kind = _draw_revision_kind(rng)
        revision = None
        if revision_kind:
            revision = _make_revision(root, topic, revision_kind, rng)
        if revision is not None:
            docs.append(revision)
            label = 1 if revision_kind == "conflict" else 0
            pairs.append(_pair(root, revision, label, revision_kind))
            if label:
                root.superseded_at = revision.issued_at
                events.append(_event(root, revision, revision.issued_at, "revision"))

        hidden = _maybe_hidden_conflict(tenant, root, topic, revision if revision_kind == "conflict" else None, counter, rng)
        if hidden is not None:
            docs.append(hidden)
            pairs.append(_pair(root, hidden, 1, "hidden_conflict"))
            if rng.random() < P_HIDDEN_CONFIRMED:
                confirmed_at = min(hidden.issued_at + pd.Timedelta(days=int(rng.integers(7, 61))), KB_END)
                events.append(_event(root, hidden, confirmed_at, "hidden_confirmed"))

        if rng.random() < P_LEXICAL_NEGATIVE:
            negative = _make_lexical_negative(tenant, root, counter, rng)
            docs.append(negative)
            pairs.append(_pair(root, negative, 0, "lexical_overlap"))

    n_positive = sum(p["label"] for p in pairs)
    n_random = int(round((NEGATIVES_PER_POSITIVE * n_positive - (len(pairs) - n_positive)) * RANDOM_NEGATIVE_OVERSAMPLE))
    pairs.extend(_random_negatives(roots, pairs, max(n_random, 0), rng))
    return docs, pairs, events


class _CodeCounter:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.value = 0

    def next(self) -> str:
        self.value += 1
        return f"{self.prefix}-{self.value:03d}"


def _topic(tenant: Tenant, key: str) -> Topic:
    return next(t for t in tenant.topics if t.key == key)


def _random_date(rng, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    span = max((end - start).days, 1)
    return start + pd.Timedelta(days=int(rng.integers(0, span)))


def _draw_values(clause, rng) -> dict:
    return {name: slot.choices[int(rng.integers(len(slot.choices)))] for name, slot in clause.slots.items()}


def _make_root(tenant: Tenant, topic: Topic, subject: str, code: str, rng) -> Doc:
    n_clauses = min(len(topic.clauses), int(rng.integers(4, 6)))
    chosen = sorted(rng.choice(len(topic.clauses), size=n_clauses, replace=False))
    clauses = [
        ClauseState(topic.clauses[i].key, int(rng.integers(len(topic.clauses[i].phrasings))), _draw_values(topic.clauses[i], rng))
        for i in chosen
    ]
    n_boiler = len(tenant.boilerplate) if tenant.key == "policy" else int(rng.integers(0, 2))
    boilerplate = list(rng.choice(tenant.boilerplate, size=min(n_boiler, len(tenant.boilerplate)), replace=False))
    return Doc(
        doc_id=f"{code}-v1",
        code=code,
        version=1,
        tenant=tenant.key,
        root_id=f"{code}-v1",
        series_id=code,
        topic=topic.key,
        subject=subject,
        department=topic.department,
        title=f"{topic.title} – {_capitalize(subject)}",
        issued_at=_random_date(rng, KB_START, ROOT_ISSUE_END),
        variant_type="root",
        clauses=clauses,
        boilerplate=boilerplate,
    )


def _add_root_citations(roots: list[Doc], rng) -> None:
    for doc in roots:
        earlier = [r for r in roots if r.department == doc.department and r.issued_at < doc.issued_at]
        if not earlier:
            continue
        n_cites = min(len(earlier), int(rng.integers(0, 3)))
        doc.cites = [earlier[i].code for i in rng.choice(len(earlier), size=n_cites, replace=False)]


def _draw_revision_kind(rng) -> str | None:
    draw = rng.random()
    if draw < P_CONFLICT_REVISION:
        return "conflict"
    if draw < P_CONFLICT_REVISION + P_PARAPHRASE_REVISION:
        return "paraphrase"
    return None


def _make_revision(root: Doc, topic: Topic, kind: str, rng) -> Doc | None:
    issued_at = root.issued_at + pd.Timedelta(days=int(rng.integers(120, 701)))
    if issued_at > KB_END:
        return None
    clauses = [ClauseState(c.key, c.phrasing, dict(c.values)) for c in root.clauses]
    changed = _change_values(clauses, topic, int(rng.integers(1, 3)), rng) if kind == "conflict" else set()
    _reword(clauses, topic, REVISION_REWORD_PROBABILITY, rng)
    _edit_structure(clauses, topic, changed, rng)
    if kind == "paraphrase":
        clauses = [clauses[i] for i in rng.permutation(len(clauses))]
    return Doc(
        doc_id=f"{root.code}-v2",
        code=root.code,
        version=2,
        tenant=root.tenant,
        root_id=root.root_id,
        series_id=root.series_id,
        topic=root.topic,
        subject=root.subject,
        department=root.department,
        title=root.title,
        issued_at=issued_at,
        variant_type=kind,
        clauses=clauses,
        boilerplate=list(root.boilerplate),
        cites=[root.code],
        revision_of=root.doc_id,
    )


def _maybe_hidden_conflict(tenant: Tenant, root: Doc, topic: Topic, conflict_revision: Doc | None, counter, rng) -> Doc | None:
    if conflict_revision is not None:
        if rng.random() >= P_HIDDEN_BEFORE_REVISION:
            return None
        latest = conflict_revision.issued_at - pd.Timedelta(days=30)
        earliest = max(root.issued_at + pd.Timedelta(days=30), conflict_revision.issued_at - pd.Timedelta(days=150))
    else:
        if rng.random() >= P_HIDDEN_OTHERWISE:
            return None
        earliest = root.issued_at + pd.Timedelta(days=60)
        latest = min(root.issued_at + pd.Timedelta(days=600), KB_END)
    if latest <= earliest:
        return None

    n_keep = min(len(root.clauses), int(rng.integers(2, 4)))
    kept = [root.clauses[i] for i in sorted(rng.choice(len(root.clauses), size=n_keep, replace=False))]
    clauses = [ClauseState(c.key, c.phrasing, dict(c.values)) for c in kept]
    if not _change_values(clauses, topic, 1, rng):
        return None
    _reword(clauses, topic, 0.5, rng)

    department = _other_department(tenant, root.department, rng)
    code = counter.next()
    return Doc(
        doc_id=f"{code}-v1",
        code=code,
        version=1,
        tenant=root.tenant,
        root_id=root.root_id,
        series_id=code,
        topic=root.topic,
        subject=root.subject,
        department=department,
        title=f"Hướng dẫn áp dụng {topic.title[0].lower() + topic.title[1:]} cho {root.subject} tại phòng {department}",
        issued_at=_random_date(rng, earliest, latest),
        variant_type="hidden_conflict",
        clauses=clauses,
        boilerplate=[],
    )


def _make_lexical_negative(tenant: Tenant, root: Doc, counter, rng) -> Doc:
    other_topics = [t for t in tenant.topics if t.key != root.topic]
    other = other_topics[int(rng.integers(len(other_topics)))]
    base = _make_root(tenant, other, root.subject, counter.next(), rng)
    root_topic = _topic(tenant, root.topic)
    base.title = f"{root_topic.title} – {_capitalize(root.subject)}: phụ lục về {other.title[0].lower() + other.title[1:]}"
    base.root_id = root.root_id
    base.variant_type = "lexical_overlap"
    base.issued_at = _random_date(rng, KB_START, KB_END)
    return base


def _edit_structure(clauses: list[ClauseState], topic: Topic, protected: set[str], rng) -> None:
    """Real revisions rarely touch only one clause: sometimes a clause is added
    or an untouched clause is dropped, for both kinds of revision alike."""
    unused = [c for c in topic.clauses if c.key not in {state.key for state in clauses}]
    if unused and rng.random() < P_REVISION_ADDS_CLAUSE:
        template = unused[int(rng.integers(len(unused)))]
        position = int(rng.integers(len(clauses) + 1))
        clauses.insert(position, ClauseState(template.key, int(rng.integers(len(template.phrasings))), _draw_values(template, rng)))
    droppable = [i for i, state in enumerate(clauses) if state.key not in protected]
    if len(clauses) > 3 and droppable and rng.random() < P_REVISION_DROPS_CLAUSE:
        del clauses[droppable[int(rng.integers(len(droppable)))]]


def _change_values(clauses: list[ClauseState], topic: Topic, n_changes: int, rng) -> set[str]:
    """Change one slot value in up to n_changes clauses; return the changed clause keys."""
    templates = {c.key: c for c in topic.clauses}
    candidates = [i for i, c in enumerate(clauses) if templates[c.key].slots]
    if not candidates:
        return set()
    changed = set()
    for i in rng.choice(candidates, size=min(n_changes, len(candidates)), replace=False):
        state = clauses[i]
        slot_name = list(templates[state.key].slots)[int(rng.integers(len(templates[state.key].slots)))]
        options = [v for v in templates[state.key].slots[slot_name].choices if v != state.values[slot_name]]
        state.values[slot_name] = options[int(rng.integers(len(options)))]
        changed.add(state.key)
    return changed


def _reword(clauses: list[ClauseState], topic: Topic, probability: float, rng) -> None:
    templates = {c.key: c for c in topic.clauses}
    for state in clauses:
        n_phrasings = len(templates[state.key].phrasings)
        if n_phrasings > 1 and rng.random() < probability:
            state.phrasing = (state.phrasing + int(rng.integers(1, n_phrasings))) % n_phrasings


def _other_department(tenant: Tenant, department: str, rng) -> str:
    others = [d for d in tenant.departments if d != department]
    return others[int(rng.integers(len(others)))]


def _pair(a: Doc, b: Doc, label: int, variant_type: str) -> dict:
    first, second = (a, b) if a.issued_at <= b.issued_at else (b, a)
    return {
        "doc_a": first.doc_id,
        "doc_b": second.doc_id,
        "tenant": a.tenant,
        "group_a": first.root_id,
        "group_b": second.root_id,
        "label": label,
        "variant_type": variant_type,
    }


def _event(a: Doc, b: Doc, confirmed_at: pd.Timestamp, source: str) -> dict:
    return {"doc_a": a.doc_id, "doc_b": b.doc_id, "tenant": a.tenant, "confirmed_at": confirmed_at, "source": source}


def _random_negatives(roots: list[Doc], existing: list[dict], n: int, rng) -> list[dict]:
    taken = {frozenset((p["doc_a"], p["doc_b"])) for p in existing}
    n_same_template = int(round(n * SAME_TEMPLATE_NEGATIVE_SHARE))
    result = []
    attempts = 0
    while len(result) < n and attempts < n * 50:
        attempts += 1
        a, b = (roots[i] for i in rng.choice(len(roots), size=2, replace=False))
        want_same_topic = len(result) < n_same_template
        if (a.topic == b.topic) != want_same_topic:
            continue
        key = frozenset((a.doc_id, b.doc_id))
        if key in taken:
            continue
        taken.add(key)
        result.append(_pair(a, b, 0, "same_template" if want_same_topic else "random"))
    return result


def render_text(doc: Doc, tenant: Tenant) -> str:
    topic = _topic(tenant, doc.topic)
    templates = {c.key: c for c in topic.clauses}
    lines = [
        f"{doc.code} – {doc.title}",
        f"Phiên bản: {doc.version} | Ngày ban hành: {doc.issued_at:%d/%m/%Y} | Đơn vị ban hành: {doc.department}",
    ]
    if doc.revision_of:
        lines.append(f"Cập nhật từ {doc.code} phiên bản 1.")
    for number, state in enumerate(doc.clauses, start=1):
        sentence = templates[state.key].phrasings[state.phrasing].format(subject=doc.subject, **state.values)
        lines.append(f"Điều {number}. {_capitalize(sentence)}")
    offset = len(doc.clauses)
    for number, sentence in enumerate(doc.boilerplate, start=offset + 1):
        lines.append(f"Điều {number}. {sentence}")
    if doc.cites and not doc.revision_of:
        lines.append("Tài liệu liên quan: " + ", ".join(doc.cites) + ".")
    return "\n".join(lines)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _docs_frame(docs: list[Doc]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        doc.text = render_text(doc, TENANTS[doc.tenant])
        row = asdict(doc)
        row["clauses"] = json.dumps([asdict(c) if not isinstance(c, dict) else c for c in doc.clauses], ensure_ascii=False)
        row["boilerplate"] = json.dumps(doc.boilerplate, ensure_ascii=False)
        row["cites"] = json.dumps(doc.cites)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["issued_at"] = pd.to_datetime(frame["issued_at"])
    frame["superseded_at"] = pd.to_datetime(frame["superseded_at"])
    return frame


def build_edges(docs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """REFERENCES from document codes in the text, SAME_SERIES between versions,
    and a sparse SAME_DEPARTMENT backbone."""
    edges = []
    by_code = {code: group.sort_values("issued_at") for code, group in docs.groupby("code")}
    for row in docs.itertuples():
        for code in set(CODE_PATTERN.findall(row.text)):
            if code == row.code and row.version == 1:
                continue
            versions = by_code.get(code)
            if versions is None:
                continue
            earlier = versions[(versions["issued_at"] < row.issued_at) & (versions["doc_id"] != row.doc_id)]
            if earlier.empty:
                continue
            edges.append((row.doc_id, earlier.iloc[-1]["doc_id"], "REFERENCES"))

    for _, group in docs.groupby("series_id"):
        ids = group.sort_values("version")["doc_id"].tolist()
        edges.extend((a, b, "SAME_SERIES") for a, b in zip(ids, ids[1:]))

    for _, group in docs.groupby(["tenant", "department"]):
        ids = group["doc_id"].tolist()
        for doc_id in ids:
            peers = [p for p in ids if p != doc_id]
            if not peers:
                continue
            for peer in rng.choice(peers, size=min(DEPARTMENT_PEERS, len(peers)), replace=False):
                edges.append((doc_id, str(peer), "SAME_DEPARTMENT"))

    frame = pd.DataFrame(edges, columns=["src", "dst", "type"])
    canonical = frame.apply(lambda r: tuple(sorted((r["src"], r["dst"]))) + (r["type"],), axis=1)
    return frame.loc[~canonical.duplicated()].reset_index(drop=True)


def main() -> None:
    from src.io_utils import write_parquet
    from src.paths import DATA_PROCESSED
    from src.settings import SEED

    kb = generate_kb(seed=SEED)
    for name, frame in kb.items():
        write_parquet(frame, DATA_PROCESSED / f"kb_{name}.parquet")
    docs, pairs = kb["docs"], kb["pairs"]
    print(f"docs={len(docs)} pairs={len(pairs)} positives={int(pairs['label'].sum())}")
    print(pairs.groupby(["tenant", "variant_type"]).size().to_string())


if __name__ == "__main__":
    main()
