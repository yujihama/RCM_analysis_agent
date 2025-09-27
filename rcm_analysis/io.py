"""ファイル入出力サポート。"""

from __future__ import annotations

import io
from typing import BinaryIO, Tuple

import pandas as pd

SUPPORTED_TYPES = ("csv", "xlsx", "xls")


def load_dataframe(file: BinaryIO, filename: str) -> pd.DataFrame:
    """拡張子に応じて DataFrame を読み込む。"""

    extension = _infer_extension(filename)
    if extension == "csv":
        return pd.read_csv(file)
    if extension in {"xlsx", "xls"}:
        return pd.read_excel(file)
    raise ValueError(f"未対応のファイル形式です: {extension}")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> Tuple[bytes, str]:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig"), "text/csv"


def _infer_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()
