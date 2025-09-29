from __future__ import annotations

import json
import os
from typing import List

from openai import OpenAI

from .types import HeaderAndLabels, LABEL_OPTIONS


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _build_messages_for_header_and_labels(sample_rows: List[List[str]]) -> List[dict]:
    system = (
        "あなたは内部監査の観点からRCM表（リスクコントロールマトリクス）の構造を理解し、"
        "ヘッダー行を特定し、各カラムに以下のラベルのいずれかを割り当てます: "
        + ", ".join(LABEL_OPTIONS)
        + "。該当しない場合は 'その他' を割り当ててください。"
    )
    user = {
        "role": "user",
        "content": (
            "以下はアップロードされた表の先頭10行です。0始まりでカウントしたヘッダー行の行番号と、"
            "データ開始行（通常はヘッダーの次の行）、ヘッダー名配列、各ヘッダー名へのラベル割り当てを、"
            "必ずJSONのみで返してください。JSONは次のスキーマに従ってください:\n\n"
            "{\n"
            "  \"header_row_index\": number,\n"
            "  \"data_start_row\": number,\n"
            "  \"header_names\": string[],\n"
            "  \"column_labels\": { [headerName: string]: string }\n"
            "}\n\n"
            "制約:\n"
            "- header_names の長さは表の列数と一致させること\n"
            "- column_labels のキーは header_names の各値と一致させること\n"
            f"- ラベルは {LABEL_OPTIONS} のいずれかのみ\n\n"
            f"先頭行サンプル(JSON):\n{json.dumps(sample_rows, ensure_ascii=False)}"
        ),
    }
    return [
        {"role": "system", "content": system},
        user,
    ]


def suggest_header_and_labels(
    api_key: str,
    model: str,
    sample_rows: List[List[str]],
) -> HeaderAndLabels:
    client = get_openai_client(api_key)
    messages = _build_messages_for_header_and_labels(sample_rows)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("LLM応答が空でした。")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM応答のJSONパースに失敗しました: {e}\ncontent={content}")

    return HeaderAndLabels.model_validate(data)


def summarize_columns(api_key: str, model: str, columns: List[str]) -> str:
    client = get_openai_client(api_key)
    prompt = (
        "以下はRCMのカラム名一覧です。フォーマットの特徴を短く要約してください。"
        "用途や文脈が分かる表現で100字程度、日本語で。\n\n"
        + "\n".join(columns)
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "監査文脈の要約アシスタント"},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content or ""


def embed_texts(api_key: str, embedding_model: str, texts: List[str]) -> List[List[float]]:
    client = get_openai_client(api_key)
    res = client.embeddings.create(model=embedding_model, input=texts)
    return [d.embedding for d in res.data]


def _build_messages_for_procedure_analysis(
    risk: str,
    control: str,
    procedure_text: str,
    prior_result: str | None,
) -> List[dict]:
    def _procedure_guidelines_compact() -> str:
        return (
            "【手続の種類（整備/運用）の基準】\n"
            "- 整備: コントロールの設計有効性。時点評価。キーワード: ウォークスルー/追跡/質問/観察/規程/マニュアル/設計/文書化。\n"
            "- 運用: 期間を通じた一貫運用。キーワード: サンプリング/テスト/突合/照合/再実施/継続的/一定期間/複数月。\n"
            "\n"
            "【証跡の量（大/中/小）の基準】\n"
            "- 大: 一連の流れを裏付ける複数種類の証憑を網羅（例: 契約書/作業報告書/請求書/入金記録 等）。\n"
            "- 中: 2～3種類の主要証憑で相互突合（例: 発注書/納品書/請求書 等）。\n"
            "- 小: 単一の証跡（例: 承認印/記録の存在 等）。\n"
            "※サンプルテストでは「1サンプルあたり」の量を判定。\n"
            "\n"
            "【難易度（難/中/易）の基準】\n"
            "- 難: 高度な専門的判断や複雑な再計算/検証/分析が必要（動詞例: 評価する/分析する/妥当性を検討する/再計算する/検証する）。\n"
            "- 中: ルールに基づく照合/突合/整合性確認/承認の有無確認。\n"
            "- 易: 閲覧/質問/観察/存在確認。\n"
        )

    def _load_guidelines_text() -> str | None:
        try:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            path = os.path.join(root, "内部統制評価手続に関する標準基準.md")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    # 安全のため長すぎる場合は一部のみ添付
                    if len(text) > 10000:
                        return text[:10000] + "\n\n...（以降は省略。必要に応じて原本を参照してください）"
                    return text
        except Exception:
            pass
        return None

    system = (
        "あなたは内部監査の専門家です。与えられた手続テキストを監査の観点で分析し、"
        "以下の基準に基づき分割および判定を行い、必ずJSONのみで返してください。\n"
        "- 手続の種類（整備/運用）\n- 証跡の量（大/中/小）\n- 難易度（難/中/易）\n"
        "分割は文意の自然な単位で行い、不要な断片は含めないでください。\n\n"
        "次の詳細基準を厳格に適用してください。\n" + _procedure_guidelines_compact()
    )
    guidelines_raw = _load_guidelines_text()

    user = {
        "role": "user",
        "content": (
            "以下の情報を元に、手続テキストを分割し、各分割について種類・証跡・難易度を判定してください。"
            "必ず次のスキーマのJSONのみで返してください。\n\n"
            "{\n"
            "  \"items\": [\n"
            "    {\n"
            "      \"procedure\": string,\n"
            "      \"evaluation_type\": \"整備\" | \"運用\",\n"
            "      \"evidence_level\": \"大\" | \"中\" | \"小\",\n"
            "      \"difficulty\": \"難\" | \"中\" | \"易\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"リスク: {risk}\n"
            f"コントロール: {control}\n"
            f"手続テキスト: {procedure_text}\n"
            f"前回の手続結果: {prior_result or ''}\n"
            "判定は@内部統制評価手続に関する標準基準に整合するようにしてください。\n\n"
            "【参考資料（標準基準の要点）】\n" + _procedure_guidelines_compact()
            + ("\n\n【参考資料（標準基準の原文抜粋）】\n" + guidelines_raw if guidelines_raw else "")
        ),
    }
    return [{"role": "system", "content": system}, user]


def analyze_procedure_with_llm(
    api_key: str,
    model: str,
    risk: str,
    control: str,
    procedure_text: str,
    prior_result: str | None,
) -> list[dict]:
    client = get_openai_client(api_key)
    messages = _build_messages_for_procedure_analysis(risk, control, procedure_text, prior_result)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    if not content:
        return []
    try:
        data = json.loads(content)
        items = data.get("items") or []
        result: list[dict] = []
        for it in items:
            p = str(it.get("procedure", "")).strip()
            et = it.get("evaluation_type", "運用")
            ev = it.get("evidence_level", "中")
            df = it.get("difficulty", "中")
            if p:
                result.append(
                    {
                        "procedure": p,
                        "evaluation_type": et,
                        "evidence_level": ev,
                        "difficulty": df,
                    }
                )
        return result
    except Exception:
        return []


