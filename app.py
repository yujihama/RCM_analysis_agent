from __future__ import annotations

import io
import os
import traceback
from typing import Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rcm_analysis.io import extract_sample_rows, compute_file_hash, read_dataframe_with_header
from rcm_analysis.llm_client import suggest_header_and_labels, summarize_columns, embed_texts
from rcm_analysis.types import HeaderAndLabels, LABEL_OPTIONS, ColumnLabel
from rcm_analysis.data_processing import apply_label_rules
from rcm_analysis.patterns import PatternDB, load_db, save_db, search_similar, upsert_pattern
from rcm_analysis.procedures import build_procedures_with_llm, procedures_to_dataframe


load_dotenv()

st.set_page_config(page_title="RCM情報抽出AIエージェント", layout="wide")
st.title("J-SOX RCM情報抽出AIエージェント (MVP)")


@st.fragment
def render_processed_csv_download(csv_bytes: bytes) -> None:
    st.download_button(
        label="CSVダウンロード",
        data=csv_bytes,
        file_name="processed.csv",
        mime="text/csv",
        key="download_processed_csv",
    )


@st.fragment
def render_procedures_csv_download(csv_bytes: bytes) -> None:
    st.download_button(
        label="手続明細CSVダウンロード",
        data=csv_bytes,
        file_name="procedures.csv",
        mime="text/csv",
        key="download_procedures_csv",
    )


def _clear_session_except(keys_to_keep: set[str]) -> None:
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep:
            del st.session_state[k]


with st.sidebar:
    st.header("設定")
    default_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")
    model = st.text_input("モデル名", value=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    embedding_model = st.text_input("埋め込みモデル名", value=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    pattern_path = st.text_input("パターンDBパス", value=os.environ.get("PATTERN_DB_PATH", "patterns.json"))


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
                    st.session_state["used_hint"] = False
                except Exception as e:
                    st.error("LLM解析でエラーが発生しました。")
                    st.exception(e)

    # ヘッダー決定後のDataFrame読み込み
    if "header_and_labels" in st.session_state:
        # セッション内のラベル案を常にソースオブトゥルースにする
        hal = HeaderAndLabels.model_validate(st.session_state["header_and_labels"])
        df = read_dataframe_with_header(
            file_bytes=file_bytes,
            filename=uploaded.name,
            header_row_index=hal.header_row_index,
            data_start_row=hal.data_start_row,
            header_names=hal.header_names,
        )
        st.session_state["df"] = df

        # フェーズ2: 特徴要約・埋め込み・類似検索（セッション済みならスキップ）
        db: PatternDB = load_db(pattern_path)
        columns = list(df.columns)
        if st.session_state.get("pattern_search_done", False):
            results = st.session_state.get("pattern_results", [])
        else:
            try:
                with st.spinner("過去パターンを検索中..."):
                    summary = summarize_columns(api_key=api_key, model=model, columns=columns)
                    [embed] = embed_texts(api_key=api_key, embedding_model=embedding_model, texts=[summary])
                    results = search_similar(db, embed, columns, top_k=1, alpha=0.8)
                # セッションへ保存（以後の再実行でスキップ）
                st.session_state["pattern_results"] = results
                st.session_state["pattern_search_done"] = True
            except Exception as e:
                results = []
                st.info("パターン検索でエラーが発生しました（スキップ）。")

        hint_text = None
        if results:
            p, total, cos, jac = results[0]
            if total > 0.75 and not st.session_state.get("used_hint", False):  # 閾値超えでヒントとして提示（初回のみ）
                hint_text = f"過去の \"{p.name or p.id}\" パターンを参考に提案を作成しました"
                st.info(hint_text)
                # 既存ルールを初期値に適用
                hal = HeaderAndLabels(
                    header_row_index=hal.header_row_index,
                    data_start_row=hal.data_start_row,
                    header_names=hal.header_names,
                    column_labels={**p.rules, **hal.column_labels},
                )
                st.session_state["header_and_labels"] = hal.model_dump()
                st.session_state["used_hint"] = True

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

        # プレビュー（フォーム送信で更新）。送信時はセッションのラベル案を更新して固定
        if submitted:
            st.session_state["header_and_labels"] = HeaderAndLabels(
                header_row_index=hal.header_row_index,
                data_start_row=hal.data_start_row,
                header_names=hal.header_names,
                column_labels=edited,
            ).model_dump()
            mapping = edited
        else:
            mapping = hal.column_labels
        preview = apply_label_rules(st.session_state["df"], mapping)
        st.subheader("プレビュー")
        st.dataframe(preview, use_container_width=True)

        # パターン名はボタン外で入力させて安定化
        pattern_name = st.text_input("パターン名（保存時に使用）", value=st.session_state.get("pattern_name", ""), key="pattern_name")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("承認"):
                st.session_state["final_mapping"] = mapping
                st.session_state["final_output"] = preview
                st.success("承認しました。CSVとしてダウンロードできます。")
                # パターン保存
                try:
                    columns = list(df.columns)
                    name = st.session_state.get("pattern_name", "")
                    summary = summarize_columns(api_key=api_key, model=model, columns=columns)
                    try:
                        [embed] = embed_texts(api_key=api_key, embedding_model=embedding_model, texts=[summary])
                    except Exception as e_embed:
                        st.warning("埋め込み生成でエラーが発生したため、列一致のみで保存します。")
                        embed = []
                    db = load_db(pattern_path)
                    upsert_pattern(db, name=name, columns=columns, summary=summary, embedding=embed, rules=mapping)
                    save_db(db, pattern_path)
                    st.toast("パターンを保存しました。")
                except Exception as e:
                    st.error("パターン保存でエラーが発生しました。")
                    st.exception(e)

        with col2:
            final_output: pd.DataFrame | None = st.session_state.get("final_output")
            if final_output is not None:
                csv_bytes = final_output.to_csv(index=False).encode("utf-8-sig")
                render_processed_csv_download(csv_bytes)

        # --- 精査機能（フェーズ3） ---
        st.markdown("---")
        st.subheader("各手続の精査と構造化")
        if st.button("手続を分割・判定する"):
            try:
                items = build_procedures_with_llm(preview, api_key=api_key, model=model)
                proc_df = procedures_to_dataframe(items)
                st.session_state["proc_df"] = proc_df
            except Exception as e:
                st.error("LLM解析でエラーが発生しました。")
                st.exception(e)
        proc_df: pd.DataFrame | None = st.session_state.get("proc_df")
        if proc_df is not None:
            st.dataframe(proc_df, use_container_width=True)
            csv_bytes = proc_df.to_csv(index=False).encode("utf-8-sig")
            render_procedures_csv_download(csv_bytes)

else:
    st.info("左のサイドバーでAPIキーを設定し、ファイルをアップロードしてください。")


