"""抽出ルール定義。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd


class ColumnRole(str, Enum):
    """RCM 内でのカラム役割。"""

    RISK = "リスク"
    CONTROL = "コントロール"
    PROCEDURE = "手続"
    OWNER = "責任者"
    FREQUENCY = "頻度"
    OTHER = "その他"

    @classmethod
    def from_label(cls, label: str) -> "ColumnRole":
        for member in cls:
            if member.value == label:
                return member
        return cls.OTHER


@dataclass
class ColumnRule:
    """単一カラムの役割定義。"""

    column: str
    role: ColumnRole


@dataclass
class CombineRule:
    """複数カラムを連結するルール。"""

    sources: List[str]
    target_name: str
    separator: str = "\n"


@dataclass
class RuleSet:
    """抽出ルール一式。"""

    column_rules: List[ColumnRule] = field(default_factory=list)
    combine_rules: List[CombineRule] = field(default_factory=list)

    @classmethod
    def from_mapping(
        cls,
        column_mapping: Mapping[str, str],
        combine_rules: Optional[Iterable[Mapping[str, str]]] = None,
    ) -> "RuleSet":
        """辞書定義からルールを作成する。"""

        column_rules = [
            ColumnRule(column=column, role=ColumnRole.from_label(role))
            for column, role in column_mapping.items()
        ]
        combine: List[CombineRule] = []
        if combine_rules:
            for rule in combine_rules:
                sources = list(rule.get("columns", []))
                target_name = str(rule.get("new_column", "結合列"))
                separator = str(rule.get("separator", "\n"))
                if not sources:
                    continue
                combine.append(
                    CombineRule(
                        sources=sources,
                        target_name=target_name,
                        separator=separator,
                    )
                )
        return cls(column_rules=column_rules, combine_rules=combine)

    def to_column_mapping(self) -> Dict[str, str]:
        return {rule.column: rule.role.value for rule in self.column_rules}

    def to_combine_mapping(self) -> List[Mapping[str, str]]:
        return [
            {
                "columns": rule.sources,
                "new_column": rule.target_name,
                "separator": rule.separator,
            }
            for rule in self.combine_rules
        ]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """ルールを適用した DataFrame を返す。"""

        processed = df.copy()
        processed = _apply_combine_rules(processed, self.combine_rules)
        ordered_columns = [rule.column for rule in self.column_rules if rule.column in processed]
        remaining = [col for col in processed.columns if col not in ordered_columns]
        return processed[ordered_columns + remaining]


def _apply_combine_rules(df: pd.DataFrame, combine_rules: Iterable[CombineRule]) -> pd.DataFrame:
    result = df.copy()
    for rule in combine_rules:
        missing = [column for column in rule.sources if column not in result.columns]
        if missing:
            continue
        combined = result[rule.sources].fillna("").agg(rule.separator.join, axis=1)
        result[rule.target_name] = combined
    return result


def summarize_roles(rules: Iterable[ColumnRule]) -> MutableMapping[str, List[str]]:
    """役割ごとにカラム名をグルーピングする。"""

    summary: MutableMapping[str, List[str]] = {}
    for rule in rules:
        summary.setdefault(rule.role.value, []).append(rule.column)
    return summary
