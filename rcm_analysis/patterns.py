"""パターン学習および参照ロジック。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from difflib import SequenceMatcher
import uuid


@dataclass(frozen=True)
class PatternRecord:
    """承認済みの抽出パターンを表現するデータ構造。"""

    pattern_id: str
    name: str
    features: Mapping[str, object]
    rules: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "features": dict(self.features),
            "rules": dict(self.rules),
        }


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class PatternRepository:
    """ローカル JSON に承認済みパターンを保存・検索するリポジトリ。"""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.storage_path = storage_path or Path("data/patterns.json")
        self._patterns: Dict[str, PatternRecord] = {}
        self._load()

    # 公開 API ------------------------------------------------------------
    def list(self) -> Sequence[PatternRecord]:
        return tuple(self._patterns.values())

    def get(self, pattern_id: str) -> Optional[PatternRecord]:
        return self._patterns.get(pattern_id)

    def upsert(
        self,
        *,
        pattern_id: Optional[str],
        name: str,
        features: Mapping[str, object],
        rules: Mapping[str, object],
    ) -> PatternRecord:
        record = PatternRecord(
            pattern_id=pattern_id or _generate_identifier(name, features),
            name=name,
            features=features,
            rules=rules,
        )
        self._patterns[record.pattern_id] = record
        self._persist()
        return record

    def find_similar(
        self,
        *,
        summary: str,
        columns: Iterable[str],
        limit: int = 1,
        threshold: float = 0.35,
    ) -> List[Tuple[PatternRecord, float]]:
        """サマリとカラム情報に基づき類似パターンを検索する。"""

        candidates: List[Tuple[PatternRecord, float]] = []
        columns_set = {str(column).lower() for column in columns}
        for record in self._patterns.values():
            raw_columns = record.features.get("columns", [])
            if isinstance(raw_columns, Mapping):
                iterable_columns: Iterable[str] = raw_columns.values()  # type: ignore[assignment]
            elif isinstance(raw_columns, (list, tuple, set)):
                iterable_columns = raw_columns  # type: ignore[assignment]
            else:
                iterable_columns = []
            record_columns = {str(column).lower() for column in iterable_columns}
            score = _similarity_score(
                summary,
                str(record.features.get("summary", "")),
                columns_set,
                record_columns,
            )
            if score >= threshold:
                candidates.append((record, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:limit]

    # 内部ユーティリティ ------------------------------------------------
    def _load(self) -> None:
        if not self.storage_path.exists():
            self._patterns = {}
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._patterns = {}
            return
        records: Dict[str, PatternRecord] = {}
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, Mapping):
                continue
            pattern_id = str(entry.get("pattern_id", ""))
            name = str(entry.get("name", pattern_id))
            features = entry.get("features", {})
            rules = entry.get("rules", {})
            if not pattern_id:
                continue
            records[pattern_id] = PatternRecord(
                pattern_id=pattern_id,
                name=name,
                features=features if isinstance(features, Mapping) else {},
                rules=rules if isinstance(rules, Mapping) else {},
            )
        self._patterns = records

    def _persist(self) -> None:
        _ensure_directory(self.storage_path)
        payload = [record.to_dict() for record in self._patterns.values()]
        self.storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_identifier(name: str, features: Mapping[str, object]) -> str:
    base = name.strip().replace(" ", "_") or "pattern"
    suffix = uuid.uuid4().hex[:8]
    return f"{base}_{suffix}"


def _similarity_score(
    query_summary: str,
    record_summary: str,
    query_columns: Iterable[str],
    record_columns: Iterable[str],
) -> float:
    """サマリ文章とカラム構成の類似度を統合したスコアを算出する。"""

    summary_ratio = SequenceMatcher(None, query_summary, record_summary).ratio()
    query_set = set(query_columns)
    record_set = set(record_columns)
    if not query_set and not record_set:
        column_ratio = 1.0
    else:
        intersection = len(query_set & record_set)
        union = len(query_set | record_set) or 1
        column_ratio = intersection / union
    return 0.6 * column_ratio + 0.4 * summary_ratio


def build_reference_payload(record: PatternRecord) -> Mapping[str, object]:
    """LLM プロンプト向けに参照情報を整形する。"""

    return {
        "pattern_id": record.pattern_id,
        "name": record.name,
        "summary": record.features.get("summary", ""),
        "rules": record.rules,
    }
