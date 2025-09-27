from __future__ import annotations

import io
import os
import traceback
from typing import Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rcm_analysis.io import extract_sample_rows, compute_file_hash, read_dataframe_with_header
from rcm_analysis.llm_client import suggest_header_and_labels
from rcm_analysis.types import HeaderAndLabels, LABEL_OPTIONS, ColumnLabel
from rcm_analysis.data_processing import apply_label_rules


load_dotenv()

st.set_page_config(page_title="RCM情報抽出AIエージェント", layout="wide")
st.title("J-SOX RCM情報抽出AIエージェント (MVP)")


def _clear_session_except(keys_to_keep: set[str]) -> None:
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep:
            del st.session_state[k]


with st.sidebar:
    st.header("設定")
    default_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")
    model = st.text_input("モデル名", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    st.caption("エラーはフォールバックせずそのまま表示します")


uploaded = st.file_uploader("CSV/Excelファイルをアップロード", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_hash = compute_file_hash(file_bytes)
    prev_hash = st.session_state.get("file_hash")
    if prev_hash != file_hash:
        _clear_session_except({"OPENAI_API_KEY"})
        st.session_state["file_hash"] = file_hash

    # 初期解析（セッションに結果が無ければLLMに投げる）
    if "header_and_labels" not in st.session_state:
        if not api_key:
            st.warning("APIキーを入力してください。")
        else:
            with st.spinner("AIがヘッダーとカラムの役割を解析中..."):
                try:
                    sample = extract_sample_rows(file_bytes, uploaded.name, max_rows=10)
                    hal: HeaderAndLabels = suggest_header_and_labels(api_key=api_key, model=model, sample_rows=sample)
                    st.session_state["header_and_labels"] = hal.model_dump()
                except Exception as e:
                    st.error("LLM解析でエラーが発生しました。")
                    st.exception(e)

    # ヘッダー決定後のDataFrame読み込み
    if "header_and_labels" in st.session_state:
        hal = HeaderAndLabels.model_validate(st.session_state["header_and_labels"])
        df = read_dataframe_with_header(
            file_bytes=file_bytes,
            filename=uploaded.name,
            header_row_index=hal.header_row_index,
            data_start_row=hal.data_start_row,
            header_names=hal.header_names,
        )
        st.session_state["df"] = df

        st.subheader("AIが提案したルール案（カラム → ラベル）")
        # 編集可能UI
        with st.form("rule_edit_form", clear_on_submit=False):
            edited: Dict[str, ColumnLabel] = {}
            cols = list(df.columns)
            grid = st.columns(3)
            for i, col in enumerate(cols):
                with grid[i % 3]:
                    default_label = hal.column_labels.get(col, "その他") if hal else "その他"
                    edited[col] = st.selectbox(
                        label=f"{col}",
                        options=LABEL_OPTIONS,
                        index=LABEL_OPTIONS.index(default_label) if default_label in LABEL_OPTIONS else LABEL_OPTIONS.index("その他"),
                        key=f"label_{i}_{col}",
                    )  # type: ignore[assignment]

            submitted = st.form_submit_button("プレビュー更新")

        # プレビュー（フォーム送信で更新）
        mapping = edited if submitted else hal.column_labels
        preview = apply_label_rules(st.session_state["df"], mapping)
        st.subheader("プレビュー")
        st.dataframe(preview, use_container_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("承認"):
                st.session_state["final_mapping"] = mapping
                st.session_state["final_output"] = preview
                st.success("承認しました。CSVとしてダウンロードできます。")

        with col2:
            final_output: pd.DataFrame | None = st.session_state.get("final_output")
            if final_output is not None:
                csv_bytes = final_output.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSVダウンロード",
                    data=csv_bytes,
                    file_name="processed.csv",
                    mime="text/csv",
                )

else:
    st.info("左のサイドバーでAPIキーを設定し、ファイルをアップロードしてください。")


