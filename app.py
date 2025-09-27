"""Streamlit アプリケーションのエントリーポイント。"""

from __future__ import annotations

import json
from typing import Dict, Optional
import streamlit as st

from rcm_analysis.data_processing import (
    apply_rules,
    build_rule_set,
    export_rules,
    generate_rule_prompt_payload,
)
from rcm_analysis.io import dataframe_to_csv_bytes, load_dataframe
from rcm_analysis.llm_client import LLMClient, LLMConfiguration, LLMError, allowed_roles
from rcm_analysis.patterns import PatternRepository, build_reference_payload

pattern_repository = PatternRepository()

st.set_page_config(page_title="RCM 情報抽出エージェント", layout="wide")

st.title("RCM 情報抽出エージェント (MVP)")
st.markdown(
    """
    単一のファイルをアップロードし、AI が提案する抽出ルールを確認・修正しながら
    CSV を生成することができます。
    """
)

uploaded_file = st.file_uploader(
    "RCM ファイルをアップロード",
    type=["csv", "xlsx", "xls"],
)

if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}
if "combine_rules" not in st.session_state:
    st.session_state.combine_rules = {}
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "pattern_summary" not in st.session_state:
    st.session_state.pattern_summary = ""
if "reference_pattern" not in st.session_state:
    st.session_state.reference_pattern = None
if "pattern_name" not in st.session_state:
    st.session_state.pattern_name = ""
if "saved_pattern_id" not in st.session_state:
    st.session_state.saved_pattern_id = None

if uploaded_file is not None:
    try:
        df = load_dataframe(uploaded_file, uploaded_file.name)
    except Exception as exc:
        st.error(f"ファイルの読み込みに失敗しました: {exc}")
    else:
        st.session_state.dataframe = df
        st.success("ファイルを読み込みました。")
        st.dataframe(df.head(20))

        with st.spinner("LLM によるカラム役割を推論中..."):
            client = LLMClient(LLMConfiguration.from_env())
            summary = client.summarize_dataframe(df)
            st.session_state.pattern_summary = summary
            similar_candidates = pattern_repository.find_similar(
                summary=summary,
                columns=df.columns,
                limit=1,
            )
            reference_payload: Optional[Dict[str, object]] = None
            if similar_candidates:
                reference_record, score = similar_candidates[0]
                st.session_state.reference_pattern = reference_record
                reference_payload = dict(build_reference_payload(reference_record))
                st.info(
                    f"過去のパターン『{reference_record.name}』(スコア {score:.2f}) を参照して提案を生成します。"
                )
            else:
                st.session_state.reference_pattern = None
            try:
                suggestions = client.suggest_column_roles(df, reference=reference_payload)
            except LLMError as exc:
                st.warning(f"LLM を利用できませんでした。ヒューリスティックな推論を利用します: {exc}")
                suggestions = client.fallback_column_roles(df)
            st.session_state.column_mapping = dict(suggestions)
            default_name = uploaded_file.name.rsplit(".", 1)[0]
            st.session_state.pattern_name = f"{default_name} パターン"
            st.session_state.saved_pattern_id = None

if st.session_state.dataframe is not None:
    df = st.session_state.dataframe
    st.subheader("カラム役割の確認")
    role_options = list(allowed_roles())
    current_mapping: Dict[str, str] = st.session_state.column_mapping or {}

    edited_mapping: Dict[str, str] = {}
    cols = st.columns([1, 1])
    for index, column in enumerate(df.columns):
        container = cols[index % 2]
        with container:
            role = current_mapping.get(column, "その他")
            edited_mapping[column] = st.selectbox(
                f"{column}",
                role_options,
                index=role_options.index(role) if role in role_options else role_options.index("その他"),
            )
    st.session_state.column_mapping = edited_mapping

    st.subheader("カラム結合ルールの設定")
    combine_rules_state: Dict[str, Dict[str, object]] = st.session_state.combine_rules or {}
    available_columns = list(df.columns)

    with st.expander("結合ルールを編集"):
        rule_names = list(combine_rules_state.keys()) or ["結合ルール1"]
        for name in rule_names:
            rule = combine_rules_state.setdefault(
                name,
                {
                    "columns": [],
                    "new_column": name,
                    "separator": "\n",
                },
            )
            st.markdown(f"#### {name}")
            selected_columns = st.multiselect(
                f"{name} で結合するカラム",
                options=available_columns,
                default=rule.get("columns", []),
                key=f"combine_columns_{name}",
            )
            rule["columns"] = selected_columns
            rule["new_column"] = st.text_input(
                "新しいカラム名",
                value=str(rule.get("new_column", name)),
                key=f"combine_new_col_{name}",
            )
            rule["separator"] = st.text_input(
                "結合時のセパレータ",
                value=str(rule.get("separator", "\n")),
                key=f"combine_sep_{name}",
            )
        if st.button("結合ルールを追加"):
            combine_rules_state[f"結合ルール{len(combine_rules_state) + 1}"] = {
                "columns": [],
                "new_column": f"新規カラム{len(combine_rules_state) + 1}",
                "separator": "\n",
            }
    st.session_state.combine_rules = combine_rules_state

    rules = build_rule_set(edited_mapping, combine_rules_state.values())
    rules_json = export_rules(rules)

    st.subheader("ルール適用結果プレビュー")
    processed_df = apply_rules(df, rules)
    st.dataframe(processed_df.head(20))

    csv_bytes, mime = dataframe_to_csv_bytes(processed_df)
    st.download_button(
        "CSV をダウンロード",
        data=csv_bytes,
        file_name="processed_rcm.csv",
        mime=mime,
    )

    st.subheader("ルール JSON")
    st.json(rules_json)

    st.subheader("プロンプト共有用情報")
    st.code(
        json.dumps(
            generate_rule_prompt_payload(
                df,
                reference=(
                    build_reference_payload(st.session_state.reference_pattern)
                    if st.session_state.reference_pattern
                    else None
                ),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )

    st.subheader("パターン承認")
    if st.session_state.reference_pattern:
        st.caption(
            f"提案の元となった参考パターン: {st.session_state.reference_pattern.name}"
        )
    pattern_name = st.text_input(
        "保存するパターン名",
        value=st.session_state.pattern_name,
        key="pattern_name_input",
    )
    st.session_state.pattern_name = pattern_name

    if st.button("承認してパターンを保存"):
        features = {
            "summary": st.session_state.pattern_summary,
            "columns": list(map(str, df.columns)),
            "row_count": len(df),
        }
        record = pattern_repository.upsert(
            pattern_id=st.session_state.saved_pattern_id,
            name=pattern_name or "名称未設定パターン",
            features=features,
            rules=rules_json,
        )
        st.session_state.saved_pattern_id = record.pattern_id
        st.success(f"パターンを保存しました (ID: {record.pattern_id})。")
else:
    st.info("ファイルをアップロードすると処理が開始されます。")
