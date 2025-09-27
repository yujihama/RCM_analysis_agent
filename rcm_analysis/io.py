from __future__ import annotations

import hashlib
from io import BytesIO, StringIO
from typing import List, Optional, Tuple

import pandas as pd


def compute_file_hash(file_bytes: bytes) -> str:
    sha = hashlib.sha256()
    sha.update(file_bytes)
    return sha.hexdigest()


def _detect_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "excel"
    return "unknown"


def extract_sample_rows(file_bytes: bytes, filename: str, max_rows: int = 10) -> List[List[str]]:
    """
    先頭 max_rows 行を文字列の二次元配列として返す。ヘッダー判定のためのサンプル入力。
    """
    kind = _detect_extension(filename)
    if kind == "csv":
        # 多少の文字コード差異に頑健にする
        text = file_bytes.decode("utf-8-sig", errors="ignore")
        df = pd.read_csv(StringIO(text), header=None, nrows=max_rows)
    elif kind == "excel":
        df = pd.read_excel(BytesIO(file_bytes), header=None, nrows=max_rows)
    else:
        # 不明拡張子はCSVとして試す
        text = file_bytes.decode("utf-8-sig", errors="ignore")
        df = pd.read_csv(StringIO(text), header=None, nrows=max_rows)

    # 文字列化
    rows: List[List[str]] = []
    for _, row in df.iterrows():
        rows.append(["" if pd.isna(v) else str(v) for v in row.tolist()])
    return rows


def read_dataframe_with_header(
    file_bytes: bytes,
    filename: str,
    header_row_index: int,
    data_start_row: Optional[int],
    header_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    LLM判定済みのヘッダー位置・列名でDataFrameを構築する。
    header_names が与えられた場合は強制的に列名に適用する。
    data_start_row が None の場合は header_row_index + 1 とみなす。
    """
    kind = _detect_extension(filename)
    start = header_row_index if header_row_index is not None else 0
    data_start = data_start_row if data_start_row is not None else (start + 1)

    if kind == "csv":
        text = file_bytes.decode("utf-8-sig", errors="ignore")
        raw = pd.read_csv(
            StringIO(text), header=None, dtype=str, keep_default_na=False
        )
    elif kind == "excel":
        raw = pd.read_excel(
            BytesIO(file_bytes), header=None, dtype=str, keep_default_na=False
        )
    else:
        text = file_bytes.decode("utf-8-sig", errors="ignore")
        raw = pd.read_csv(
            StringIO(text), header=None, dtype=str, keep_default_na=False
        )

    # ヘッダー行の取得
    header_series = raw.iloc[start]
    cols = header_series.astype(str).tolist()
    if header_names and len(header_names) == len(cols):
        cols = header_names

    # データ部分の抽出
    df = raw.iloc[data_start:].copy()
    df.columns = cols
    df.reset_index(drop=True, inplace=True)
    return df


