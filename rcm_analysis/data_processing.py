from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd

from .types import ColumnLabel, LABEL_OPTIONS


MAIN_LABEL_ORDER: List[ColumnLabel] = [
    "リスク",
    "コントロール",
    "手続",
    "前回の手続結果",
    "その他",
]


def _columns_by_label(column_labels: Dict[str, ColumnLabel], df: pd.DataFrame) -> Dict[ColumnLabel, List[str]]:
    present = set(df.columns.astype(str).tolist())
    by_label: Dict[ColumnLabel, List[str]] = {label: [] for label in MAIN_LABEL_ORDER}
    for col, label in column_labels.items():
        if col in present:
            by_label[label].append(col)
    return by_label


def _jsonify_row(row: pd.Series, cols: List[str]) -> str:
    payload = {col: (None if pd.isna(row[col]) or row[col] == "" else row[col]) for col in cols}
    return json.dumps(payload, ensure_ascii=False)


def apply_label_rules(df: pd.DataFrame, column_labels: Dict[str, ColumnLabel]) -> pd.DataFrame:
    """
    ラベルに基づき、出力用の5カラム（リスク/コントロール/手続/前回の手続結果/その他）を作成する。
    同一ラベルに複数カラムが紐づく場合、そのセルは {元カラム名: 値} のJSON文字列にする。
    """
    by_label = _columns_by_label(column_labels, df)

    out = pd.DataFrame(index=df.index)
    for label in MAIN_LABEL_ORDER:
        cols = by_label.get(label, [])
        if len(cols) == 0:
            out[label] = ""
        elif len(cols) == 1:
            out[label] = df[cols[0]]
        else:
            out[label] = df.apply(lambda r: _jsonify_row(r, cols), axis=1)

    return out


