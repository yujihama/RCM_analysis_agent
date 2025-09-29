from __future__ import annotations

import uuid
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from .llm_client import analyze_procedure_with_llm


EvidenceLevel = Literal["大", "中", "小"]
DifficultyLevel = Literal["難", "中", "易"]
EvaluationType = Literal["整備", "運用"]


class ProcedureItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_row: int
    risk: str
    control: str
    procedure: str
    prior_result: Optional[str] = None
    evidence_level: EvidenceLevel
    difficulty: DifficultyLevel
    evaluation_type: EvaluationType
    exclude: bool = False


def build_procedures_with_llm(
    df_out: pd.DataFrame,
    api_key: str,
    model: str,
) -> List[ProcedureItem]:
    items: List[ProcedureItem] = []
    for idx, row in df_out.iterrows():
        risk = str(row.get("リスク", "") or "")
        control = str(row.get("コントロール", "") or "")
        proc_cell = row.get("手続", "")
        prior = str(row.get("前回の手続結果", "") or "")
        text = proc_cell if isinstance(proc_cell, str) else str(proc_cell)

        llm_items = analyze_procedure_with_llm(
            api_key=api_key,
            model=model,
            risk=risk,
            control=control,
            procedure_text=text,
            prior_result=(prior if prior != "" else None),
        )

        if not llm_items:
            raise ValueError("LLMが手続分割・判定結果を返しませんでした。")

        for s_idx, it in enumerate(llm_items):
            item = ProcedureItem(
                id=f"{idx}-{s_idx}",
                source_row=int(idx),
                risk=risk,
                control=control,
                procedure=str(it.get("procedure", "")),
                prior_result=(prior if prior != "" else None),
                evidence_level=str(it.get("evidence_level", "中")),
                difficulty=str(it.get("difficulty", "中")),
                evaluation_type=str(it.get("evaluation_type", "運用")),
                exclude=False,
            )
            items.append(item)

    return items


def procedures_to_dataframe(items: List[ProcedureItem]) -> pd.DataFrame:
    data = [
        {
            "ID": it.id,
            "行": it.source_row,
            "リスク": it.risk,
            "コントロール": it.control,
            "手続": it.procedure,
            "前回の手続結果": it.prior_result or "",
            "手続の種類": it.evaluation_type,
            "証跡の量": it.evidence_level,
            "難易度": it.difficulty,
        }
        for it in items
    ]
    return pd.DataFrame(data)


