from __future__ import annotations

import json
import re
import uuid
from typing import Iterable, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from .llm_client import analyze_procedure_with_llm


EvidenceLevel = Literal["大", "中", "小"]
DifficultyLevel = Literal["難", "中", "易"]
EvaluationType = Literal["整備", "運用"]


HARD_KEYS = ["評価", "分析", "妥当性", "再計算", "検証"]
MEDIUM_KEYS = ["照合", "突合", "整合性", "承認の有無"]
EASY_KEYS = ["閲覧", "質問", "観察", "存在を確認", "存在確認"]

SEIBI_KEYS = ["ウォークスルー", "追跡", "質問", "観察", "規程", "マニュアル", "設計", "文書化"]
UNYOU_KEYS = ["サンプリング", "サンプル", "テスト", "突合", "照合", "再実施", "継続的", "一定期間", "複数月"]

DOC_KEYS = [
    "契約書",
    "作業報告書",
    "請求書",
    "入金記録",
    "稟議書",
    "現物確認",
    "伝票",
    "台帳",
    "調整表",
    "承認記録",
]

EVIDENCE_BIG_HINTS = ["網羅", "一連", "全て", "すべて", "包括", "複数の証憑", "複数の証跡", "複数種類"]
EVIDENCE_MED_HINTS = ["2点", "3点", "2つ", "3つ", "突合", "照合"]
EVIDENCE_SMALL_HINTS = ["承認印", "存在のみ", "閲覧", "質問", "観察"]


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


_SEGMENT_SPLIT_RE = re.compile(r"[。\n；;]|・")


def segment_procedures(text: str) -> List[str]:
    if not text:
        return []
    # JSON文字列の場合もある（複数カラムがJSON化されているケース）
    segments: List[str] = []
    raw = text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            raw = "。".join(str(v) for v in parsed.values() if v)
    except Exception:
        pass
    for frag in _SEGMENT_SPLIT_RE.split(str(raw)):
        frag = frag.strip()
        if len(frag) >= 2:
            segments.append(frag)
    return segments


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    t = str(text)
    return any(k in t for k in keywords)


def estimate_evaluation_type(text: str) -> EvaluationType:
    score_seibi = sum(1 for k in SEIBI_KEYS if k in text)
    score_unyou = sum(1 for k in UNYOU_KEYS if k in text)
    if score_unyou > score_seibi:
        return "運用"
    if score_unyou < score_seibi:
        return "整備"
    # デフォルトは運用寄り（テスト文脈が多いため）
    return "運用"


def estimate_difficulty(text: str) -> DifficultyLevel:
    if _contains_any(text, HARD_KEYS):
        return "難"
    if _contains_any(text, MEDIUM_KEYS):
        return "中"
    if _contains_any(text, EASY_KEYS):
        return "易"
    # 不明時は中
    return "中"


def estimate_evidence_level(text: str) -> EvidenceLevel:
    doc_hits = sum(1 for k in DOC_KEYS if k in text)
    if _contains_any(text, EVIDENCE_BIG_HINTS) or doc_hits >= 4:
        return "大"
    if _contains_any(text, EVIDENCE_MED_HINTS) or 2 <= doc_hits <= 3:
        return "中"
    if _contains_any(text, EVIDENCE_SMALL_HINTS) or doc_hits <= 1:
        return "小"
    return "中"


def build_procedures(df_out: pd.DataFrame) -> List[ProcedureItem]:
    items: List[ProcedureItem] = []
    for idx, row in df_out.iterrows():
        risk = str(row.get("リスク", "") or "")
        control = str(row.get("コントロール", "") or "")
        proc_cell = row.get("手続", "")
        prior = str(row.get("前回の手続結果", "") or "")
        text = proc_cell if isinstance(proc_cell, str) else str(proc_cell)
        segments = segment_procedures(text)
        for s_idx, seg in enumerate(segments):
            ev_type = estimate_evaluation_type(seg)
            diff = estimate_difficulty(seg)
            ev_level = estimate_evidence_level(seg)
            item = ProcedureItem(
                id=f"{idx}-{s_idx}",
                source_row=int(idx),
                risk=risk,
                control=control,
                procedure=seg,
                prior_result=(prior if prior != "" else None),
                evidence_level=ev_level,
                difficulty=diff,
                evaluation_type=ev_type,
                exclude=False,
            )
            items.append(item)
    return items


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


