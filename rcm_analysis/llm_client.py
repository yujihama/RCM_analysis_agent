from __future__ import annotations

import json
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
        temperature=0.1,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("LLM応答が空でした。")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM応答のJSONパースに失敗しました: {e}\ncontent={content}")

    return HeaderAndLabels.model_validate(data)


