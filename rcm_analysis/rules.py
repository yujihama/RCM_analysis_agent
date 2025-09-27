"""抽出ルール定義。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd


class ColumnRole(str, Enum):
    """RCM 内でのカラム役割。"""

    RISK = "リスク"
    CONTROL = "コントロール"
    PROCEDURE = "手続"
    PREVIOUS_PROCEDURE_RESULT = "前回の手続結果"

    @classmethod
    def from_label(cls, label: str) -> "ColumnRole":
        for member in cls:
            if member.value == label:
                return member
        # 該当する役割が見つからない場合は、特別な値としてリスクを返す（実際の処理で除外するため）
        return cls.RISK


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
        """ルールを適用した DataFrame を返す。指定された役割のカラムのみを結合して返す。"""

        processed = df.copy()

        # 役割ごとにカラムをグループ化
        role_columns = {}
        for rule in self.column_rules:
            if rule.column in processed.columns:
                role = rule.role.value
                if role not in role_columns:
                    role_columns[role] = []
                role_columns[role].append(rule.column)

        # 各役割のカラムをJSON形式で結合（指定された4つの役割のみ）
        result_data = {}
        allowed_roles = {"リスク", "コントロール", "手続", "前回の手続結果"}

        for role, columns in role_columns.items():
            if role in allowed_roles and columns:
                # 複数のカラムがある場合はJSON形式で結合
                if len(columns) > 1:
                    # 各行のJSONオブジェクトを作成
                    def create_json_string(row):
                        data = {}
                        for col in columns:
                            value = row[col]
                            if pd.isna(value):
                                data[col] = ""
                            else:
                                data[col] = str(value)
                        return json.dumps(data, ensure_ascii=False)

                    combined = processed[columns].apply(create_json_string, axis=1)
                    result_data[role] = combined
                else:
                    # 1つのカラムしかない場合はそのまま（数値も文字列に変換）
                    result_data[role] = processed[columns[0]].fillna("").astype(str)

        return pd.DataFrame(result_data)


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
