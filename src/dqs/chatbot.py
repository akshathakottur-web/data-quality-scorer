"""
chatbot.py

Conversational interface for asking questions about data quality
findings and requesting fixes, powered by Groq (free, no billing).

Design principles (project standards):
- The LLM only ever sees a SUMMARY of the actual computed report,
  never raw/full data -- keeps context small and prevents hallucinated
  numbers, since every fact it can reference was computed by our
  deterministic detectors, not guessed.
- The LLM can only take action by calling one of our pre-defined,
  tested fix functions (AVAILABLE_FIXES). It never writes or executes
  arbitrary code against the dataset.
- Every fix returns a human-readable confirmation message so the user
  always sees exactly what changed.
- Row-deletion is additionally constrained in code (not just prompt
  instructions) to rows our detectors actually flagged -- see
  _get_flaggable_row_indices / the check in chat_turn.
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
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
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
            "description": "Remove specific rows by index. Only rows already identified as anomalies, outliers, or suspicious labels in the analysis can actually be removed -- any other index will be rejected.",
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


def _get_flaggable_row_indices(all_reports: dict) -> set:
    """
    The complete set of row indices our detectors have actually flagged
    as problematic. remove_rows_by_index is only allowed to touch rows
    in this set -- this is the code-level enforcement of "only remove
    rows already identified as suspicious," not just a prompt instruction
    the LLM could ignore or hallucinate around.
    """
    flaggable = set()

    flaggable.update(all_reports.get("anomalies", {}).get("anomalous_indices", []) or [])
    flaggable.update(all_reports.get("outliers_iforest", {}).get("outlier_indices", []) or [])
    flaggable.update(all_reports.get("label_issues", {}).get("suspicious_indices", []) or [])
    flaggable.update(all_reports.get("duplicates", {}).get("duplicate_row_indices", []) or [])
    flaggable.update(all_reports.get("row_quality", {}).get("problem_row_indices", []) or [])

    return flaggable


def build_system_prompt(df, profile: dict, all_reports: dict, scoring: dict) -> str:
    """
    Build grounded context from the ACTUAL computed analysis -- never
    from the LLM's assumptions. This is what keeps answers accurate
    to this specific dataset.
    """
    dataset_context = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": {"rows": profile["n_rows"], "columns": profile["n_cols"]},
        "guessed_target": profile.get("guessed_target"),
        "id_like_columns": profile.get("id_like_cols"),
    }

    row_quality = all_reports.get("row_quality", {})
    validity = all_reports.get("validity", {})

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

        # --- validity (email/date/range) findings ---
        "validity_applicable": validity.get("applicable", False),
        "invalid_emails_by_column": {
            col: v.get("n_invalid") for col, v in validity.get("email_validity", {}).items()
        } if validity.get("applicable") else None,
        "invalid_dates_by_column": {
            col: v.get("n_invalid") for col, v in validity.get("date_validity", {}).items()
        } if validity.get("applicable") else None,
        "out_of_range_by_column": {
            col: v.get("n_invalid") for col, v in validity.get("range_validity", {}).items()
        } if validity.get("applicable") else None,

        # --- row-level quality (this is the AUTHORITATIVE overall score) ---
        "pct_rows_with_at_least_one_problem": row_quality.get("pct_problem_rows"),
        "row_quality_breakdown": row_quality.get("breakdown"),
    }

    return f"""You are the Data Quality Assistant for the "Automatic Data Quality
Scoring System" project. You help the user understand and fix issues
in THEIR SPECIFIC dataset, based ONLY on the analysis below.

DATASET STRUCTURE:
{json.dumps(dataset_context, indent=2, default=str)}

DATA QUALITY ANALYSIS RESULTS:
{json.dumps(quality_summary, indent=2, default=str)}

Note: "overall_score" is a row-level score -- the % of rows with ZERO
flagged problems across every check (missing values, duplicates,
invalid emails/dates, out-of-range values, statistical outliers). If
asked why the score is low, "row_quality_breakdown" and
"pct_rows_with_at_least_one_problem" are the most direct explanation.

STRICT RULES:
1. Only reference column names that appear in DATASET STRUCTURE above.
   If a user mentions a column that doesn't exist, tell them it wasn't
   found and list the actual columns instead.
2. Only reference numbers/findings that appear in the ANALYSIS RESULTS
   above. Never invent statistics, scores, or counts.
3. If the user asks a QUESTION, answer concisely using this data.
4. If the user clearly asks you to FIX something, call the matching
   tool. Do not call a tool for vague or ambiguous requests -- ask a
   clarifying question instead.
5. If a requested fix doesn't map to any available tool (e.g. fixing
   a malformed email or an invalid date -- there is currently no tool
   for these), say so plainly instead of guessing or calling the wrong tool.
6. For remove_rows_by_index specifically: only ever pass indices that
   appear in anomalous_row_indices, isolation_forest_outlier_indices,
   suspicious_label_indices, or the duplicate row indices above. Never
   invent or guess an index. If the user asks to remove a row that
   isn't in one of those lists, explain that it hasn't been flagged by
   any detector and ask them to confirm they still want it removed
   manually (which this assistant cannot do).
7. After any fix, briefly state what changed and note that the
   analysis above has been refreshed to reflect it.
8. If the user asks to download or export the updated dataset, tell them
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
    flaggable_indices = _get_flaggable_row_indices(all_reports)

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

    # IMPORTANT:
    # Start cleaning from a COPY so the original dataframe is not
    # directly modified by any fix function.
    updated_df = df.copy()

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

            # FIX ONLY FOR drop_duplicate_rows:
            # This function takes no arguments, so ignore any malformed
            # empty arguments returned by the model.
            if fn_name == "drop_duplicate_rows":
                fn_args = {}

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

            # --- Code-level safety check: row deletion is restricted to
            # rows our detectors actually flagged, regardless of what the
            # LLM decided to pass. This cannot be bypassed by prompting.
            if fn_name == "remove_rows_by_index":
                requested = set(fn_args.get("indices", []))
                not_flagged = requested - flaggable_indices

                if not_flagged:
                    reply_parts.append(
                        f"⚠️ Row(s) {sorted(not_flagged)} haven't been flagged by any "
                        f"detector (anomaly, outlier, suspicious label, or duplicate), "
                        f"so I won't remove them automatically. If you're sure, please "
                        f"remove them yourself, or ask about what's actually flagged."
                    )
                    continue

            try:
                updated_df, fix_message = fix_fn(updated_df, **fn_args)

            except Exception as e:
                reply_parts.append(f"⚠️ Couldn't apply that fix ({fn_name}): {e}")
                continue

            reply_parts.append(f"✅ {fix_message}")
            fix_applied = True

    if not reply_parts:
        reply_parts.append(
            "I didn't quite catch that — try asking about a specific finding or requesting a fix."
        )

    return {
        "reply": "\n\n".join(reply_parts),
        "updated_df": updated_df,
        "fix_applied": fix_applied,
    }