"""LLM クライアントモジュール。

OpenAI API を利用して DataFrame の構造を推論するための補助関数を提供します。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai パッケージが未インストールの環境を考慮
    OpenAI = None  # type: ignore


class LLMError(RuntimeError):
    """LLM 呼び出し時のエラーを表す例外。"""


@dataclass(frozen=True)
class LLMConfiguration:
    """LLM 呼び出しに必要な設定情報。"""

    api_key: Optional[str] = None
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfiguration":
        """環境変数から設定を読み込む。"""

        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        )


class LLMClient:
    """LLM を用いてカラム役割の推論を行うクライアント。"""

    def __init__(self, config: Optional[LLMConfiguration] = None) -> None:
        self.config = config or LLMConfiguration.from_env()
        if self.config.api_key and OpenAI is None:
            raise LLMError(
                "openai パッケージがインストールされていないため LLM を利用できません。"
            )
        self._client: Optional[OpenAI] = None
        if self.config.api_key and OpenAI is not None:
            self._client = OpenAI(api_key=self.config.api_key)

    def suggest_column_roles(
        self,
        df: pd.DataFrame,
        sample_size: int = 5,
        reference: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, str]:
        """DataFrame のカラム役割を推論する。

        Parameters
        ----------
        df: pd.DataFrame
            対象のデータフレーム。
        sample_size: int, default 5
            プロンプトに含めるサンプル行数。
        """

        if df.empty:
            return {}
        if not self.config.api_key or self._client is None:
            raise LLMError("API キーが設定されていないため LLM を利用できません。")

        payload = self._build_prompt(df, sample_size, reference)
        try:
            response = self._client.responses.create(
                model=self.config.model,
                temperature=self.config.temperature,
                input=payload,
            )
        except Exception as exc:  # pragma: no cover - 実行時に API エラーをそのまま通知
            raise LLMError("LLM からの応答を取得できませんでした。") from exc

        raw_text = response.output_text  # type: ignore[attr-defined]
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMError("LLM から期待する JSON を取得できませんでした。") from exc

        if not isinstance(parsed, Mapping):
            raise LLMError("LLM からの応答形式が不正です。")
        return {str(key): str(value) for key, value in parsed.items()}

    def _build_prompt(
        self,
        df: pd.DataFrame,
        sample_size: int,
        reference: Optional[Mapping[str, object]],
    ) -> str:
        columns = list(df.columns)
        sample_records = df.head(sample_size)
        prompt = {
            "instruction": "与えられた表の各カラムが RCM (Risk Control Matrix) のどの役割を担うかを推論してください。",
            "columns": columns,
            "samples": sample_records.to_dict(orient="records"),
            "output_format": {
                "column_name": "role",
            },
            "allowed_roles": _allowed_roles(),
        }
        if reference:
            prompt["reference_pattern"] = reference
        return json.dumps(prompt, ensure_ascii=False)

    def _heuristic_guess(self, df: pd.DataFrame) -> Mapping[str, str]:
        """API が利用できない場合の簡易推論。"""

        keywords = {
            "リスク": ["risk", "リスク"],
            "コントロール": ["control", "統制", "コントロール"],
            "手続": ["procedure", "手続", "activity"],
        }
        roles = {}
        for column in df.columns:
            lower = str(column).lower()
            assigned = "その他"
            for role, patterns in keywords.items():
                if any(keyword in lower for keyword in patterns):
                    assigned = role
                    break
            roles[str(column)] = assigned
        return roles

    def fallback_column_roles(self, df: pd.DataFrame) -> Mapping[str, str]:
        """フォールバック用にヒューリスティック推論を公開する。"""

        return self._heuristic_guess(df)

    def summarize_dataframe(self, df: pd.DataFrame, sample_size: int = 5) -> str:
        """LLM を用いて DataFrame の特徴を要約する。"""

        if df.empty:
            return ""
        if not self.config.api_key or self._client is None:
            raise LLMError("API キーが設定されていないため LLM を利用できません。")

        payload = {
            "instruction": "与えられた RCM テーブルの構造と特徴を 3 文以内で要約してください。",
            "columns": list(df.columns),
            "samples": df.head(sample_size).to_dict(orient="records"),
        }
        try:
            response = self._client.responses.create(
                model=self.config.model,
                temperature=self.config.temperature,
                input=json.dumps(payload, ensure_ascii=False),
            )
            return response.output_text.strip()  # type: ignore[attr-defined]
        except Exception as exc:
            raise LLMError("LLM からの要約取得に失敗しました。") from exc

    def _fallback_summary(self, df: pd.DataFrame) -> str:
        if len(df.columns) == 0:
            return f"列情報なし。行数: {len(df)}"
        head_columns = ", ".join(map(str, df.columns[:5]))
        additional = "" if len(df.columns) <= 5 else f" ほか {len(df.columns) - 5} 列"
        return f"主な列: {head_columns}{additional}. 行数: {len(df)}"


def _allowed_roles() -> List[str]:
    return [
        "リスク",
        "コントロール",
        "手続",
        "責任者",
        "頻度",
        "その他",
    ]


def allowed_roles() -> Iterable[str]:
    """外部公開用の役割一覧。"""

    return tuple(_allowed_roles())
