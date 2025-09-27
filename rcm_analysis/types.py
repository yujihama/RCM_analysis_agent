from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, Field, field_validator


ColumnLabel = Literal["リスク", "コントロール", "手続", "前回の手続結果", "その他"]

LABEL_OPTIONS: List[ColumnLabel] = [
    "リスク",
    "コントロール",
    "手続",
    "前回の手続結果",
    "その他",
]


class HeaderAndLabels(BaseModel):
    """
    LLMが返すヘッダー行位置とカラム名、各カラムへのラベル付け結果。
    header_row_index は 0 始まりの行番号（先頭行が 0）。
    data_start_row はデータ開始行（多くの場合 header_row_index + 1）。
    """

    header_row_index: int = Field(..., ge=0)
    data_start_row: int = Field(..., ge=0)
    header_names: List[str]
    column_labels: Dict[str, ColumnLabel]

    @field_validator("header_names")
    @classmethod
    def _no_empty_names(cls, v: List[str]) -> List[str]:
        cleaned = [name if (name is not None and str(name).strip() != "") else f"col_{i}"
                   for i, name in enumerate(v)]
        return cleaned

    @field_validator("column_labels")
    @classmethod
    def _coerce_labels(cls, v: Dict[str, str]) -> Dict[str, ColumnLabel]:
        coerced: Dict[str, ColumnLabel] = {}
        for k, raw in v.items():
            label = str(raw).strip()
            if label not in LABEL_OPTIONS:
                label = "その他"
            coerced[k] = label  # type: ignore[assignment]
        return coerced


