"""
chatbot.py

Conversational interface for asking questions about data quality
findings and requesting fixes, powered by Groq (free, no billing).

Design principles (project standards):
- The LLM only ever sees a SUMMARY of the actual computed report,
  never raw/full data — keeps context small and prevents hallucinated
  numbers, since every fact it can reference was computed by our
  deterministic detectors, not guessed.
- The LLM can only take action by calling one of our pre-defined,
  tested fix functions (AVAILABLE_FIXES). It never writes or executes
  arbitrary code against the dataset.
- Every fix returns a human-readable confirmation message so the user
  always sees exactly what changed.
"""

import json
import inspect
import streamlit as st
from groq import Groq
from src.dqs.fixes import AVAILABLE_FIXES

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

MODEL_NAME = "openai/gpt-oss-20b"  # fast + free tier; swap to "llama-3.3-70b-versatile" for stronger reasoning

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "drop_columns",
            "description": "Remove one or more columns entirely from the dataset. Use for constant columns, redundant/correlated columns, or columns the user explicitly wants removed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drop_duplicate_rows",
            "description": "Remove exact duplicate rows from the dataset.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "impute_missing",
            "description": "Fill missing values in a specific column using a chosen strategy (median/mean for numeric, mode for categorical).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "strategy": {"type": "string", "enum": ["median", "mean", "mode"]},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cap_outliers_iqr",
            "description": "Cap (winsorize) outliers in a numeric column to IQR bounds instead of deleting rows.",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string"}},
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_rows_by_index",
            "description": "Remove specific rows by index. Use only for rows already identified as anomalies or suspicious labels in the analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["indices"],
            },
        },
    },
]


def build_system_prompt(df, profile: dict, all_reports: dict, scoring: dict) -> str:
    """
    Build grounded context from the ACTUAL computed analysis — never
    from the LLM's assumptions. This is what keeps answers accurate
    to this specific dataset.
    """
    dataset_context = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": {"rows": profile["n_rows"], "columns": profile["n_cols"]},
        "guessed_target": profile.get("guessed_target"),
    }

    quality_summary = {
        "overall_score": scoring["overall_score"],
        "status": scoring["status"],
        "individual_scores": scoring["individual_scores"],
        "missing_values_by_column": all_reports.get("missing_values", {}).get("column_missing"),
        "duplicate_rows": all_reports.get("duplicates", {}).get("n_duplicate_rows"),
        "constant_columns": all_reports.get("constant_columns", {}).get("constant_columns"),
        "correlated_pairs": all_reports.get("correlated_features", {}).get("correlated_pairs"),
        "outlier_columns_iqr": all_reports.get("outliers_iqr", {}).get("column_outliers"),
        "isolation_forest_outlier_indices": all_reports.get("outliers_iforest", {}).get("outlier_indices"),
        "class_imbalance": all_reports.get("class_imbalance", {}),
        "anomalous_row_indices": all_reports.get("anomalies", {}).get("anomalous_indices"),
        "suspicious_label_indices": all_reports.get("label_issues", {}).get("suspicious_indices"),
        "suspicious_leakage_features": all_reports.get("feature_leakage", {}).get("suspicious_features"),
    }

    return f"""You are the Data Quality Assistant for the "Automatic Data Quality
Scoring System" project. You help the user understand and fix issues
in THEIR SPECIFIC dataset, based ONLY on the analysis below.

DATASET STRUCTURE:
{json.dumps(dataset_context, indent=2, default=str)}

DATA QUALITY ANALYSIS RESULTS:
{json.dumps(quality_summary, indent=2, default=str)}

STRICT RULES:
1. Only reference column names that appear in DATASET STRUCTURE above.
   If a user mentions a column that doesn't exist, tell them it wasn't
   found and list the actual columns instead.
2. Only reference numbers/findings that appear in the ANALYSIS RESULTS
   above. Never invent statistics, scores, or counts.
3. If the user asks a QUESTION, answer concisely using this data.
4. If the user clearly asks you to FIX something, call the matching
   tool. Do not call a tool for vague or ambiguous requests — ask a
   clarifying question instead.
5. If a requested fix doesn't map to any available tool, say so plainly
   instead of guessing.
6. After any fix, briefly state what changed and note that re-running
   the analysis will show the updated score.
7. If the user asks to download or export the updated dataset, tell them
   to use the "Download updated CSV" button below the chat, which always
   reflects the current state of the dataset after any fixes applied here.
"""


def chat_turn(user_message: str, conversation_history: list,
              df, profile: dict, all_reports: dict, scoring: dict) -> dict:
    """
    Process one conversational turn.

    Returns
    -------
    dict with:
        - reply: text to display
        - updated_df: dataframe, modified if a fix was applied
        - fix_applied: bool
    """
    system_prompt = build_system_prompt(df, profile, all_reports, scoring)

    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )

    choice = response.choices[0].message
    updated_df = df
    fix_applied = False
    reply_parts = []

    if choice.content:
        reply_parts.append(choice.content)

    if choice.tool_calls:
        for call in choice.tool_calls:
            fn_name = call.function.name
            try:
                fn_args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                reply_parts.append(f"⚠️ Couldn't parse arguments for {fn_name}.")
                continue

            fix_fn = AVAILABLE_FIXES.get(fn_name)
            if not fix_fn:
                reply_parts.append(f"⚠️ Unknown fix requested: {fn_name}")
                continue

            # Build a case-insensitive lookup: lowercase name -> actual column name
            column_lookup = {c.lower(): c for c in df.columns}

            col_keys = [k for k in fn_args if "column" in k]
            invalid_cols = []

            for key in col_keys:
                value = fn_args[key]
                targets = value if isinstance(value, list) else [value]
                corrected = []

                for t in targets:
                    match = column_lookup.get(t.lower())

                    if match:
                        corrected.append(match)
                    else:
                        invalid_cols.append(t)

                fn_args[key] = (
                    corrected
                    if isinstance(value, list)
                    else (corrected[0] if corrected else value)
                )

            if invalid_cols:
                reply_parts.append(
                    f"⚠️ Column(s) {invalid_cols} not found in the dataset — no changes made."
                )
                continue

            # Validate fn_args against the real function signature before calling it,
            # so a malformed tool call from the LLM never crashes the app.
            sig = inspect.signature(fix_fn)
            valid_params = set(sig.parameters.keys()) - {"df"}
            required_params = {
                name for name, param in sig.parameters.items()
                if param.default is inspect.Parameter.empty and name != "df"
            }

            unexpected = set(fn_args.keys()) - valid_params
            missing = required_params - set(fn_args.keys())

            if unexpected:
                reply_parts.append(
                    f"⚠️ The assistant tried to use an invalid argument {list(unexpected)} "
                    f"for '{fn_name}' — no changes made. Try rephrasing your request."
                )
                continue

            if missing:
                reply_parts.append(
                    f"⚠️ Missing required info ({list(missing)}) to run '{fn_name}' — "
                    f"please specify it, e.g. which column."
                )
                continue

            try:
                updated_df, fix_message = fix_fn(updated_df, **fn_args)
                reply_parts.append(f"✅ {fix_message}")
                fix_applied = True
            except Exception as e:
                reply_parts.append(
                    f"⚠️ Couldn't apply the fix '{fn_name}': {str(e)}"
                )

    if not reply_parts:
        reply_parts.append(
            "I didn't quite catch that — try asking about a specific finding or requesting a fix."
        )

    return {
        "reply": "\n\n".join(reply_parts),
        "updated_df": updated_df,
        "fix_applied": fix_applied,
    }