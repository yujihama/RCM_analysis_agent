"""データ処理ロジック。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from .llm_client import allowed_roles
from .rules import ColumnRule, ColumnRole, CombineRule, RuleSet


def generate_rule_prompt_payload(
    df: pd.DataFrame,
    sample_size: int = 5,
    reference: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    """LLM に渡すプロンプト用の情報を組み立てる。"""

    payload = {
        "instruction": "RCM テーブルのカラム役割を判定してください。",
        "columns": list(df.columns),
        "samples": df.head(sample_size).to_dict(orient="records"),
        "allowed_roles": list(allowed_roles()),
    }
    if reference:
        payload["reference_pattern"] = reference
    return payload


def build_rule_set(column_mapping: Mapping[str, str], combine_rules: Iterable[Mapping[str, str]]) -> RuleSet:
    return RuleSet.from_mapping(column_mapping=column_mapping, combine_rules=combine_rules)


def apply_rules(df: pd.DataFrame, rules: RuleSet) -> pd.DataFrame:
    return rules.apply(df)


def export_rules(rules: RuleSet) -> Mapping[str, object]:
    return {
        "column_rules": [
            {
                "column": rule.column,
                "role": rule.role.value,
            }
            for rule in rules.column_rules
        ],
        "combine_rules": [asdict(rule) for rule in rules.combine_rules],
    }


def ensure_column_rules(df: pd.DataFrame, mapping: Mapping[str, str]) -> Sequence[ColumnRule]:
    return [
        ColumnRule(column=column, role=ColumnRole.from_label(mapping.get(column, "その他")))
        for column in df.columns
    ]


def infer_combine_rules(selection: Mapping[str, Mapping[str, object]]) -> Iterable[CombineRule]:
    combines: list[CombineRule] = []
    for key, payload in selection.items():
        sources = payload.get("columns", [])
        if not isinstance(sources, (list, tuple)) or not sources:
            continue
        target_name = str(payload.get("new_column", key))
        separator = str(payload.get("separator", "\n"))
        combines.append(CombineRule(sources=list(sources), target_name=target_name, separator=separator))
    return combines
