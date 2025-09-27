from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field


class Pattern(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    columns: List[str] = []
    summary: str = ""
    embedding: List[float] = []
    rules: Dict[str, str] = {}
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())


class PatternDB(BaseModel):
    version: int = 1
    patterns: List[Pattern] = []


def load_db(path: str) -> PatternDB:
    if not os.path.exists(path):
        return PatternDB()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return PatternDB.model_validate(raw)
    except Exception:
        # 壊れている場合は新規作成
        return PatternDB()


def save_db(db: PatternDB, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db.model_dump(), f, ensure_ascii=False, indent=2)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def search_similar(
    db: PatternDB,
    query_embedding: List[float],
    query_columns: List[str],
    top_k: int = 3,
    alpha: float = 0.8,
) -> List[Tuple[Pattern, float, float, float]]:
    """
    類似度 = alpha * cos(embedding) + (1 - alpha) * jaccard(columns)
    戻り値: (pattern, total_score, cos, jaccard)
    """
    scored: List[Tuple[Pattern, float, float, float]] = []
    for p in db.patterns:
        cos = _cosine_similarity(query_embedding, p.embedding)
        jac = _jaccard(query_columns, p.columns)
        total = alpha * cos + (1 - alpha) * jac
        scored.append((p, total, cos, jac))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def upsert_pattern(
    db: PatternDB,
    name: str,
    columns: List[str],
    summary: str,
    embedding: List[float],
    rules: Dict[str, str],
    match_threshold: float = 0.97,
) -> Pattern:
    # ほぼ同一（列集合一致かつ高い埋め込み類似）なら更新、それ以外は新規
    best = search_similar(db, embedding, columns, top_k=1, alpha=0.7)
    if best:
        (p, total, cos, jac) = best[0]
        if jac > 0.95 and cos > match_threshold:
            p.name = name or p.name
            p.columns = columns
            p.summary = summary
            p.embedding = embedding
            p.rules = rules
            p.updated_at = time.time()
            return p

    p = Pattern(name=name or "")
    p.columns = columns
    p.summary = summary
    p.embedding = embedding
    p.rules = rules
    db.patterns.append(p)
    return p


