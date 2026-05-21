import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import shutil

st.set_page_config(page_title="Micron Streamlit Dummy", layout="wide")

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'Datasets')

now = datetime.now().strftime("%d-%m-%Y")

if "page" not in st.session_state:
    st.session_state.page = "upload"
if "overview_shift" not in st.session_state:
    st.session_state.overview_shift = "total"
if "df" not in st.session_state:
    st.session_state.df = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None
if "ai_summary_key" not in st.session_state:
    st.session_state.ai_summary_key = None
if "actions_summary" not in st.session_state:
    st.session_state.actions_summary = None
if "actions_summary_key" not in st.session_state:
    st.session_state.actions_summary_key = None
if "util_threshold" not in st.session_state:
    st.session_state.util_threshold = 95
if "_thresh_slider" not in st.session_state:
    st.session_state._thresh_slider = 95
if "_thresh_input" not in st.session_state:
    st.session_state._thresh_input = 95
if "overview_machine" not in st.session_state:
    st.session_state.overview_machine = "All"
if "show_dataset" not in st.session_state:
    st.session_state.show_dataset = False
if "ai_summary_cache" not in st.session_state:
    st.session_state.ai_summary_cache = {}
if "actions_summary_cache" not in st.session_state:
    st.session_state.actions_summary_cache = {}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def clear_dataset_folder():
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    for item in os.listdir(DATASET_FOLDER):
        item_path = os.path.join(DATASET_FOLDER, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            st.warning(f"Could not remove {item}: {e}")

def read_uploaded_dataset(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    if filename.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")

    raise ValueError("Unsupported file type. Please upload a CSV, XLSX, or XLS file.")

def metric_box(title, value, desc=""):
    desc_html = desc if desc else "&nbsp;"

    st.markdown(f"""
        <div style="
            background-color: #1e2130;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 5px solid #4a9eff;
            min-height: 118px;
            box-sizing: border-box;
        ">
            <div style="font-size: 13px; color: #aaaaaa; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{value}</div>
            <div style="font-size: 11px; color: #777777; margin-top: 4px; min-height: 14px;">
                {desc_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

def colored_metric(title, value, desc="", color="#ffffff"):
    desc_html = desc if desc else "&nbsp;"

    st.markdown(f"""
        <div style="
            background-color: #1e2130;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 5px solid #4a9eff;
            min-height: 118px;
            box-sizing: border-box;
        ">
            <div style="font-size: 13px; color: {color}; margin-bottom: 4px; font-weight: bold;">
                {title}
            </div>
            <div style="font-size: 28px; font-weight: bold; color: #ffffff;">
                {value}
            </div>
            <div style="font-size: 11px; color: #777777; margin-top: 4px; min-height: 14px;">
                {desc_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

def html_metric(title, value_html, desc=""):
    desc_html = desc if desc else "&nbsp;"

    st.markdown(f"""
        <div style="
            background-color: #1e2130;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 5px solid #4a9eff;
            min-height: 118px;
            box-sizing: border-box;
        ">
            <div style="font-size: 13px; color: #aaaaaa; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 28px; font-weight: bold;">{value_html}</div>
            <div style="font-size: 11px; color: #777777; margin-top: 4px; min-height: 14px;">
                {desc_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

def downtime_group_box(total_downtime, fleet_repair, fleet_pm):
    html = f"""
<div style="
    background-color:#151824;
    border-radius:14px;
    border:1px solid #2c3142;
    margin-top:10px;
    overflow:hidden;
">

    <div style="
        background-color:#1e2130;
        padding:20px;
        border-left:5px solid #4a9eff;
        min-height:105px;
        box-sizing:border-box;
        text-align:center;
    ">
        <div style="font-size:13px; color:#aaaaaa; margin-bottom:8px;">
            Total Downtime
        </div>
        <div style="font-size:30px; font-weight:bold; color:#ffffff;">
            {total_downtime} min
        </div>
    </div>

    <div style="
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:0;
        border-top:1px solid #2c3142;
    ">

        <div style="
            background-color:#1e2130;
            padding:20px;
            border-left:5px solid #FF0000;
            border-right:1px solid #2c3142;
            min-height:110px;
            box-sizing:border-box;
            text-align:center;
        ">
            <div style="font-size:13px; color:#FF0000; margin-bottom:8px; font-weight:bold;">
                Total Repair Time
            </div>
            <div style="font-size:28px; font-weight:bold; color:#ffffff;">
                {fleet_repair} min
            </div>
            <div style="font-size:11px; color:#777777; margin-top:8px;">
                WAIT_REPAIR + IN_REPAIR
            </div>
        </div>

        <div style="
            background-color:#1e2130;
            padding:20px;
            border-left:5px solid #FFC0CB;
            min-height:110px;
            box-sizing:border-box;
            text-align:center;
        ">
            <div style="font-size:13px; color:#FFC0CB; margin-bottom:8px; font-weight:bold;">
                Total PM Time
            </div>
            <div style="font-size:28px; font-weight:bold; color:#ffffff;">
                {fleet_pm} min
            </div>
            <div style="font-size:11px; color:#777777; margin-top:8px;">
                WAIT_PM + IN_PM
            </div>
        </div>

    </div>
</div>
"""
    st.html(html)

def get_pct_color(pct):
    if pct >= 80:
        return "#2ecc71"
    elif pct >= 50:
        return "#f1c40f"
    else:
        return "#e74c3c"

def get_util_color(pct, threshold):
    return "#2ecc71" if pct >= threshold else "#e74c3c"

def render_threshold_sidebar():
    st.markdown("## 🎯 Overall Utilization Threshold")

    st.session_state.setdefault("util_threshold", 95)
    st.session_state.setdefault("_thresh_slider", int(st.session_state["util_threshold"]))
    st.session_state.setdefault("_thresh_input", int(st.session_state["util_threshold"]))

    def _sync_from_slider():
        val = int(st.session_state["_thresh_slider"])
        st.session_state["_thresh_input"] = val
        st.session_state["util_threshold"] = val

    def _sync_from_input():
        val = max(0, min(100, int(st.session_state["_thresh_input"])))
        st.session_state["_thresh_input"] = val
        st.session_state["_thresh_slider"] = val
        st.session_state["util_threshold"] = val

    st.slider(
        "Threshold (%)",
        min_value=0,
        max_value=100,
        step=1,
        value=int(st.session_state["_thresh_slider"]),
        key="_thresh_slider",
        on_change=_sync_from_slider,
    )

    st.number_input(
        "Manual input",
        min_value=0,
        max_value=100,
        step=1,
        value=int(st.session_state["_thresh_input"]),
        key="_thresh_input",
        on_change=_sync_from_input,
    )

    st.markdown(
        f"""
        <div style="
            background-color:#1e2130;
            border-radius:12px;
            padding:16px 18px;
            margin-top:10px;
            border-left:5px solid #4a9eff;
            display:flex;
            flex-direction:column;
            gap:8px;
            align-items:center;
        ">
            <div style="color:#2ecc71; font-weight:bold; font-size:14px;">
                🟢 ≥ {st.session_state['util_threshold']}%
            </div>
            <div style="color:#e74c3c; font-weight:bold; font-size:14px;">
                🔴 &lt; {st.session_state['util_threshold']}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def build_summary_context(filtered_df):
    fdf = filtered_df.copy()
    fdf["Status"] = fdf["Status"].astype(str).str.strip()
    fdf["Machine_ID"] = fdf["Machine_ID"].astype(str).str.strip()

    total_duration = fdf["Duration_Min"].sum()
    up_duration = fdf[fdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()

    line_util_pct = round((up_duration / total_duration) * 100, 1) if total_duration > 0 else 0

    downtime_df = fdf[fdf["Status"] != "UP_PRODUCT"].copy()
    total_downtime_min = int(downtime_df["Duration_Min"].sum())

    # Machine availability/utilization lines
    avail_rows = []

    for machine in sorted(fdf["Machine_ID"].dropna().unique()):
        mdf = fdf[fdf["Machine_ID"] == machine]
        m_total = mdf["Duration_Min"].sum()
        m_up = mdf[mdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()
        m_downtime = mdf[mdf["Status"] != "UP_PRODUCT"]["Duration_Min"].sum()

        m_util = round((m_up / m_total) * 100, 1) if m_total > 0 else 0
        m_down_pct = round((m_downtime / m_total) * 100, 1) if m_total > 0 else 0

        avail_rows.append({
            "Machine_ID": machine,
            "Utilization": m_util,
            "Downtime_Pct": m_down_pct,
            "Downtime_Min": int(m_downtime)
        })

    avail_df = pd.DataFrame(avail_rows)

    if not avail_df.empty:
        avail_df = avail_df.sort_values("Utilization", ascending=True)

        avail_lines = "\n".join([
            f"- {row['Machine_ID']}: {row['Utilization']}% utilization, {row['Downtime_Min']} min downtime"
            for _, row in avail_df.iterrows()
        ])

        bottleneck_lines = "\n".join([
            f"- {row['Machine_ID']}: {row['Downtime_Min']} min downtime, {row['Utilization']}% utilization"
            for _, row in avail_df.sort_values("Downtime_Min", ascending=False).head(3).iterrows()
        ])
    else:
        avail_lines = "- No machine availability data available"
        bottleneck_lines = "- No bottleneck data available"

    # Top downtime reasons
    reason_series = downtime_df["Downtime_Reason"].dropna().astype(str).str.strip()
    reason_series = reason_series[reason_series != ""]
    top_reasons = reason_series.value_counts().head(5)

    reasons_lines = "\n".join([
        f"- {reason}: {count} occurrence(s)"
        for reason, count in top_reasons.items()
    ]) if not top_reasons.empty else "- No downtime reasons recorded"

    # Status downtime context
    status_downtime = downtime_df.groupby("Status")["Duration_Min"].sum().sort_values(ascending=False)

    downtime_context = "\n".join([
        f"- {status}: {int(minutes)} min"
        for status, minutes in status_downtime.items()
    ]) if not status_downtime.empty else "- No downtime status recorded"

    return {
        "line_util_pct": line_util_pct,
        "total_downtime_min": total_downtime_min,
        "avail_lines": avail_lines,
        "bottleneck_lines": bottleneck_lines,
        "reasons_lines": reasons_lines,
        "downtime_context": downtime_context
    }

def call_ollama(prompt):
    import requests

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 500
        }
    }

    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(10, 120)
        )

        if resp.status_code != 200:
            return f"""
❌ Ollama API failed.

Status code: {resp.status_code}

Error:
{resp.text}
"""

        data = resp.json()

        if "response" not in data:
            return f"❌ Ollama returned an unexpected response: {data}"

        return data["response"].strip()

    except requests.exceptions.ConnectTimeout:
        return "❌ Could not connect to Ollama. Please check whether Ollama is running."

    except requests.exceptions.ReadTimeout:
        return "❌ Ollama took too long to respond. Try a smaller model such as llama3.2:3b."

    except requests.exceptions.ConnectionError:
        return """
❌ Could not connect to Ollama.

Please make sure Ollama is running locally:
1. Open Command Prompt
2. Run: ollama serve
3. In another Command Prompt, run: ollama run llama3.2:3b
"""

    except Exception as e:
        return f"❌ Error contacting Ollama: {e}"

def build_ai_summary_prompt(filtered_df, shift_label, machine_filter="All"):
    summary = build_summary_context(filtered_df)

    if machine_filter != "All":
        scope = f"{machine_filter} only, {shift_label}"
        bullet1_instruction = f"Bullet 1: How did {machine_filter} perform? Mention its utilization percentage and total downtime."
        bullet2_instruction = f"Bullet 2: What were the main reasons {machine_filter} lost time? Refer only to the downtime reasons provided."
        bullet3_instruction = f"Bullet 3: Give one specific next-shift action for {machine_filter}."
    else:
        scope = f"All machines, {shift_label}"
        bullet1_instruction = "Bullet 1: Which machine had the lowest utilization and what was its downtime?"
        bullet2_instruction = "Bullet 2: What were the top downtime reasons across all machines?"
        bullet3_instruction = "Bullet 3: Give one specific, actionable recommendation for the next shift."

    prompt = f"""
You are a manufacturing performance analyst.

Based only on the metrics below, write a concise dashboard summary with exactly 3 bullet points.

SCOPE OF THIS REPORT:
{scope}

METRICS SUMMARY
---------------

Overall Line Utilization:
{summary["line_util_pct"]}%

Machine Utilization:
{summary["avail_lines"]}

Top Bottleneck Machines:
{summary["bottleneck_lines"]}

Top Downtime Reasons:
{summary["reasons_lines"]}

Total Downtime:
{summary["total_downtime_min"]} minutes

Downtime by Status:
{summary["downtime_context"]}

Write exactly 3 bullet points based on these requirements:

Bullet 1 requirement:
{bullet1_instruction}

Bullet 2 requirement:
{bullet2_instruction}

Bullet 3 requirement:
{bullet3_instruction}

Rules:
- Output exactly 3 bullet points only.
- Each bullet point must start with •
- Each bullet point should be 1 to 2 sentences only.
- Do not use sub-bullets.
- Do not add introduction or conclusion.
- Do not invent data.
- Only use the numbers provided above.
- Only mention machines within the scope defined above.
- Use plain English.
"""
    return prompt

def build_next_actions_prompt(filtered_df, shift_label, machine_filter="All"):
    fdf = filtered_df.copy()
    fdf["Status"] = fdf["Status"].astype(str).str.strip()
    fdf["Machine_ID"] = fdf["Machine_ID"].astype(str).str.strip()

    total_duration = fdf["Duration_Min"].sum()

    wait_repair_min = int(fdf[fdf["Status"] == "WAIT_REPAIR"]["Duration_Min"].sum())
    wait_pm_min = int(fdf[fdf["Status"] == "WAIT_PM"]["Duration_Min"].sum())
    in_repair_min = int(fdf[fdf["Status"] == "IN_REPAIR"]["Duration_Min"].sum())

    wait_repair_pct = round((wait_repair_min / total_duration) * 100, 1) if total_duration > 0 else 0
    wait_pm_pct = round((wait_pm_min / total_duration) * 100, 1) if total_duration > 0 else 0

    downtime_df = fdf[fdf["Status"] != "UP_PRODUCT"].copy()
    total_downtime_min = int(downtime_df["Duration_Min"].sum())

    # Top downtime contributors by machine
    machine_downtime = downtime_df.groupby("Machine_ID")["Duration_Min"].sum().sort_values(ascending=False)

    top_downtime_machines = "\n".join([
        f"- {machine}: {int(duration)} min downtime"
        for machine, duration in machine_downtime.head(5).items()
    ]) if not machine_downtime.empty else "- No downtime contributors recorded"

    # Top downtime reasons
    reason_series = downtime_df["Downtime_Reason"].dropna().astype(str).str.strip()
    reason_series = reason_series[reason_series != ""]
    top_reasons = reason_series.value_counts().head(5)

    top_reasons_lines = "\n".join([
        f"- {reason}: {count} occurrence(s)"
        for reason, count in top_reasons.items()
    ]) if not top_reasons.empty else "- No downtime reasons recorded"

    # Downtime by status
    status_downtime = downtime_df.groupby("Status")["Duration_Min"].sum().sort_values(ascending=False)

    status_lines = "\n".join([
        f"- {status}: {int(minutes)} min"
        for status, minutes in status_downtime.items()
    ]) if not status_downtime.empty else "- No downtime status recorded"

    # Spare part signals
    spare_keywords = ["part", "parts", "spare", "material", "component", "stock"]
    spare_df = downtime_df[
        downtime_df["Downtime_Reason"].fillna("").astype(str).str.lower().str.contains(
            "|".join(spare_keywords),
            regex=True
        )
    ]

    spare_min = int(spare_df["Duration_Min"].sum()) if not spare_df.empty else 0

    spare_reasons = spare_df["Downtime_Reason"].dropna().astype(str).str.strip()
    spare_reasons = spare_reasons[spare_reasons != ""]

    spare_lines = "\n".join([
        f"- {reason}: {count} occurrence(s)"
        for reason, count in spare_reasons.value_counts().head(5).items()
    ]) if not spare_reasons.empty else "- No spare parts shortage signals found"

    if machine_filter != "All":
        scope = f"{machine_filter} only, {shift_label}"
        action1_instruction = f"Action 1: What should the next shift do first for {machine_filter} based on the biggest downtime issue?"
        action2_instruction = f"Action 2: What should be checked for {machine_filter} to reduce WAIT_REPAIR, WAIT_PM, or IN_REPAIR time?"
        action3_instruction = f"Action 3: What preventive follow-up should be done for {machine_filter} before the next shift ends?"
    else:
        scope = f"All machines, {shift_label}"
        action1_instruction = "Action 1: Which bottleneck machine or downtime area should the next shift prioritize first?"
        action2_instruction = "Action 2: What should the team do to reduce the top downtime reasons?"
        action3_instruction = "Action 3: What preventive check or coordination should be done before the next shift ends?"

    prompt = f"""
You are a manufacturing shift supervisor.

Based only on the metrics below, write exactly 3 practical next-shift actions.

SCOPE OF THIS REPORT:
{scope}

METRICS SUMMARY
---------------

Total Downtime:
{total_downtime_min} minutes

Downtime by Status:
{status_lines}

Top Downtime Machines:
{top_downtime_machines}

Top Downtime Reasons:
{top_reasons_lines}

WAIT_REPAIR:
{wait_repair_min} minutes, {wait_repair_pct}% of selected shift time

WAIT_PM:
{wait_pm_min} minutes, {wait_pm_pct}% of selected shift time

IN_REPAIR:
{in_repair_min} minutes

Spare Parts Signals:
{spare_lines}

Spare Parts Related Downtime:
{spare_min} minutes

Write exactly 3 bullet points based on these requirements:

Action 1 requirement:
{action1_instruction}

Action 2 requirement:
{action2_instruction}

Action 3 requirement:
{action3_instruction}

Rules:
- Output exactly 3 bullet points only.
- Each bullet point must start with one single • symbol.
- Do not write two bullet symbols like • •.
- Each bullet point must be an action, not just an observation.
- Each bullet point should be 1 to 2 sentences only.
- Do not use sub-bullets.
- Do not add introduction or conclusion.
- Do not invent data.
- Only use the numbers provided above.
- Only mention machines within the scope defined above.
- Use plain English.
"""
    return prompt

def render_ai_summary_section(summary_key, prompt_fn, *prompt_args):
    st.divider()
    st.markdown("### 🤖 AI Summary")

    if st.button("✨ Generate AI Summary", key=f"ai_btn_{summary_key}"):
        if summary_key in st.session_state.ai_summary_cache:
            st.session_state.ai_summary = st.session_state.ai_summary_cache[summary_key]
            st.session_state.ai_summary_key = summary_key
        else:
            with st.spinner("Generating summary with Ollama..."):
                prompt = prompt_fn(*prompt_args)
                result = call_ollama(prompt)
                st.session_state.ai_summary = result
                st.session_state.ai_summary_key = summary_key
                st.session_state.ai_summary_cache[summary_key] = result

    if st.session_state.ai_summary and st.session_state.ai_summary_key == summary_key:
        raw_text = st.session_state.ai_summary.strip()

        lines = [
            l.strip()
            for l in raw_text.split("\n")
            if l.strip().startswith("•")
        ]

        if not lines:
            parts = [p.strip() for p in raw_text.split("•") if p.strip()]
            lines = ["• " + p for p in parts]

        cleaned_lines = []

        for line in lines[:3]:
            line = line.strip()

            # Remove repeated bullet symbols such as "• • text"
            while line.startswith("• •"):
                line = "•" + line[3:].strip()

            # Ensure each line starts with exactly one bullet
            line = line.lstrip("•").strip()
            line = "• " + line

            cleaned_lines.append(line)

        lines = cleaned_lines

        if lines:
            bullet_html = "<br>".join([
                f'<div style="margin-bottom:8px;">{line}</div>'
                for line in lines
            ])
        else:
            bullet_html = st.session_state.ai_summary

        st.markdown(f"""
            <div style="
                background:#1e2130;
                border-radius:10px;
                padding:16px 20px;
                margin-top:8px;
                border-left:5px solid #4a9eff;
            ">
                <span style="color:#e0e0e0; font-size:14px; line-height:1.6;">
                    {bullet_html}
                </span>
            </div>
        """, unsafe_allow_html=True)

def render_actions_next_shift_section(summary_key, prompt_fn, *prompt_args):
    st.divider()
    st.markdown("### ✅ Actions for Next Shift")

    if st.button("✅ Generate Actions for Next Shift", key=f"actions_btn_{summary_key}"):
        if summary_key in st.session_state.actions_summary_cache:
            st.session_state.actions_summary = st.session_state.actions_summary_cache[summary_key]
            st.session_state.actions_summary_key = summary_key
        else:
            with st.spinner("Generating actions with Ollama..."):
                prompt = prompt_fn(*prompt_args)
                result = call_ollama(prompt)
                st.session_state.actions_summary = result
                st.session_state.actions_summary_key = summary_key
                st.session_state.actions_summary_cache[summary_key] = result

    if (
        st.session_state.actions_summary
        and st.session_state.actions_summary_key == summary_key
    ):
        action_lines = [
            l.strip()
            for l in st.session_state.actions_summary.split("\n")
            if l.strip().startswith("•")
        ]

        if not action_lines:
            action_lines = [
                "• " + l.strip()
                for l in st.session_state.actions_summary.split("•")
                if l.strip()
            ]

        if action_lines:
            actions_html = "<br>".join([
                f'<div style="margin-bottom:8px;">{line}</div>'
                for line in action_lines[:3]
            ])

            st.markdown(f"""
                <div style="
                    background:#1e2130;
                    border-radius:10px;
                    padding:16px 20px;
                    margin-top:8px;
                    border-left:5px solid #2ecc71;
                ">
                    <span style="color:#e0e0e0; font-size:14px; line-height:1.6;">
                        {actions_html}
                    </span>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div style="
                    background:#1e2130;
                    border-radius:10px;
                    padding:16px 20px;
                    margin-top:8px;
                    border-left:5px solid #2ecc71;
                ">
                    <span style="color:#e0e0e0; font-size:14px; line-height:1.6;">
                        {st.session_state.actions_summary}
                    </span>
                </div>
            """, unsafe_allow_html=True)

# ── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────────
if st.session_state.df is not None and st.session_state.page != "upload":
    with st.sidebar:
        st.markdown("## 🗂️ Navigation")

        if st.button(
            "📋 Overview",
            use_container_width=True,
            type="primary" if st.session_state.page == "overview" else "secondary"
        ):
            st.session_state.page = "overview"
            st.rerun()

        st.divider()

        render_threshold_sidebar()

# ── PAGE: OVERVIEW ─────────────────────────────────────────────────────────────
if st.session_state.page == "overview":
    df = st.session_state.df
    df = df.copy()
    df["Machine_ID"] = df["Machine_ID"].str.strip()
    df["Status"] = df["Status"].str.strip()
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

    st.title("FabSight - Shift Performance Dashboard")
    st.write("By Uptime Guardians")
    st.markdown(now)
    
    # ── SHIFT + MACHINE SELECTORS ────────────────────────────────────────────
    shift_options = {
        "All": "total",
        "Day": "day",
        "Night": "night",
    }

    current_shift_label = next(
        label for label, value in shift_options.items()
        if value == st.session_state.overview_shift
    )

    selector_col1, selector_col2, selector_col3 = st.columns([1, 1, 3])

    with selector_col1:
        selected_shift_label = st.selectbox(
            "Select Shift",
            options=list(shift_options.keys()),
            index=list(shift_options.keys()).index(current_shift_label),
            key="overview_shift_select",
        )

    st.session_state.overview_shift = shift_options[selected_shift_label]
    ov_active = st.session_state.overview_shift

    # Apply shift filter first
    hour = df["Start_Time"].dt.hour

    if ov_active == "day":
        ov_df = df[(hour >= 7) & (hour < 19)].copy()
    elif ov_active == "night":
        ov_df = df[(hour < 7) | (hour >= 19)].copy()
    else:
        ov_df = df.copy()

    # Machine selector is based on filtered shift data
    machine_options = ["All"] + sorted(
        ov_df["Machine_ID"].dropna().unique().tolist()
    )

    if st.session_state.overview_machine not in machine_options:
        st.session_state.overview_machine = "All"

    with selector_col2:
        selected_machine = st.selectbox(
            "Select Machine",
            options=machine_options,
            index=machine_options.index(st.session_state.overview_machine),
            key="overview_machine_select",
        )

    st.session_state.overview_machine = selected_machine

    # Apply machine filter after shift filter
    if selected_machine != "All":
        ov_df = ov_df[ov_df["Machine_ID"] == selected_machine].copy()

    st.write("")

    ov_shift_label = {
        "total": "All Shifts",
        "day":   "Day Shift (07:00 – 19:00)",
        "night": "Night Shift (19:00 – 07:00)",
    }[ov_active]

    if selected_machine != "All":
        ov_shift_label = f"{ov_shift_label} · {selected_machine}"

    total_machines = ov_df["Machine_ID"].nunique()
    total_dur = ov_df["Duration_Min"].sum()
    fleet_up = ov_df[ov_df["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()
    fleet_util = round((fleet_up / total_dur) * 100) if total_dur > 0 else 0
    fleet_repair = int(ov_df[ov_df["Status"].isin(["WAIT_REPAIR", "IN_REPAIR"])]["Duration_Min"].sum())
    fleet_idle = int(ov_df[ov_df["Status"] == "IDLE"]["Duration_Min"].sum())
    fleet_pm = int(ov_df[ov_df["Status"].isin(["WAIT_PM", "IN_PM"])]["Duration_Min"].sum())

    st.markdown(f"#### Machine Summary · {ov_shift_label}")
    # ── Additional Fleet Metrics ────────────────────────────────────────────────
    machines_running = ov_df[ov_df["Status"] == "UP_PRODUCT"]["Machine_ID"].nunique()
    total_downtime = int(
        ov_df[
            ~ov_df["Status"].isin(["UP_PRODUCT", "IDLE"])
        ]["Duration_Min"].sum()
    )

    STATUS_COLORS = {
        "UP_PRODUCT": "#008000",
        "IDLE":       "#5F5FFF",
        "WAIT_REPAIR":"#FF0000",
        "IN_REPAIR":  "#FF0000",
        "WAIT_PM":    "#FFC0CB",
        "IN_PM":      "#FFC0CB",
    }

    # Row 1: Individual boxes, centered
    spacer_left, fc1, fc2, fc_idle, spacer_right = st.columns([0.5, 1, 1, 1, 0.5])

    with fc1:
        if selected_machine != "All":
            machine_status = "Active" if fleet_util > 0 else "Inactive"
            machine_status_color = "#2ecc71" if fleet_util > 0 else "#e74c3c"

            html_metric(
                "Machine Status",
                f'<span style="color:{machine_status_color};">{machine_status}</span>',
                f"Utilization: {fleet_util}%"
            )

        else:
            running_pct = (machines_running / total_machines) * 100 if total_machines > 0 else 0
            running_color = "#2ecc71" if running_pct >= 50 else "#e74c3c"

            html_metric(
                "Active Machines",
                f'<span style="color:{running_color};">{machines_running}</span> '
                f'<span style="color:#ffffff;">out of {total_machines}</span>',
                ""
            )

    with fc2:
        colored_metric(
            "Average Utilization",
            f"{fleet_util}%",
            f"UP_PRODUCT across {total_machines} machines",
            get_util_color(fleet_util, st.session_state.util_threshold)
        )

    with fc_idle:
        metric_box(
            f'<span style="color:{STATUS_COLORS["IDLE"]};">Total Idle Time</span>',
            f'{fleet_idle} min',
            "IDLE status across all machines"
        )

    st.write("")

    # Row 2: Bigger grouped downtime box
    group_left, group_mid, group_right = st.columns([0.75, 2.5, 0.75])

    with group_mid:
        downtime_group_box(
            total_downtime,
            fleet_repair,
            fleet_pm
        )

    st.divider()

# ── UTILIZATION CALC ─────────────────────────────────────────
    ov_df["Machine_ID"] = ov_df["Machine_ID"].astype(str).str.strip()
    machines = sorted(ov_df["Machine_ID"].dropna().unique().tolist())

    target_df = pd.DataFrame({
        "Machine_ID": ["CMP-01", "CVD-01", "DIFF-01", "ETCH-01", "IMP-01", "LITHO-01"],
        "Target": [95, 86, 85, 96, 89, 88]
    })


    util_list = []

    for machine in machines:
        mdf = ov_df[ov_df["Machine_ID"] == machine]

        total = mdf["Duration_Min"].sum()
        up = mdf[mdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()

        util = round((up / total) * 100) if total > 0 else 0

        util_list.append({
            "Machine_ID": machine,
            "Utilization": util
        })

    util_df = pd.DataFrame(util_list)
    chart_df = target_df.merge(util_df, on="Machine_ID", how="left")
    chart_df["Utilization"] = chart_df["Utilization"].fillna(0)
    
    greens = chart_df[chart_df["Utilization"] >= chart_df["Target"]]
    reds = chart_df[chart_df["Utilization"] < chart_df["Target"]]

    # Alternate green/red rows
    rows = []

    for g, r in zip(greens.to_dict("records"), reds.to_dict("records")):
        rows.append(g)
        rows.append(r)

    # Add leftovers if unequal counts
    longer = greens if len(greens) > len(reds) else reds

    for extra in longer.iloc[len(rows)//2:].to_dict("records"):
        rows.append(extra)

    chart_df = pd.DataFrame(rows)
    
    if selected_machine != "All":
        bar_colors = [
            "#00ff6a" if (m == selected_machine and u >= t)
            else "#ff1900" if m == selected_machine
            else "rgba(0,0,0,0)"   # fully transparent (hidden)
            for m, u, t in zip(
                chart_df["Machine_ID"],
                chart_df["Utilization"],
                chart_df["Target"]
            )
        ]

        text_vals = [
            f"{u}%" if m == selected_machine else ""
            for m, u in zip(chart_df["Machine_ID"], chart_df["Utilization"])
        ]
    else:
        bar_colors = [
            "#00ff6a" if u >= t else "#ff1900"
            for u, t in zip(chart_df["Utilization"], chart_df["Target"])
        ]
        text_vals = [f"{u}%" for u in chart_df["Utilization"]]

# ── PLOT ──────────────────────────────────────────────────────
    st.markdown("#### Machine Utilization vs Target")
    fig = go.Figure()
    
    
    # Utilization bars
    fig.add_trace(go.Bar(
        x=chart_df["Machine_ID"],
        y=chart_df["Utilization"],
        marker=dict(
            color=bar_colors,
            line=dict(color="#00ff6a", width=0)
        ),
        text=text_vals,
        textposition="outside",
        name="Utilization",
        customdata=chart_df[["Target"]],
        hovertemplate=(
            "<b>Machine:</b> %{x}<br>"
            "<b>Date:</b> " + now + "<br>"
            "<b>Utilization:</b> %{y}%<br>"
            "<b>Target:</b> %{customdata[0]}%<br>"
            "<extra></extra>"
        )
    ))
    
    if selected_machine == "All":
        target_x = chart_df["Machine_ID"]
        target_y = chart_df["Target"]
    else:
        target_x = [selected_machine]
        target_y = chart_df[chart_df["Machine_ID"] == selected_machine]["Target"]

    fig.add_trace(go.Scatter(
        x=target_x,
        y=target_y,
        mode="markers",
        marker=dict(
            color="#f1c40f",
            size=90,
            symbol="line-ew",
            line=dict(width=2, color="#f1c40f")
        ),
        name="Target",
        hovertemplate=(
            "<b>Machine:</b> %{x}<br>"
            "<b>Date:</b> " + now + "<br>"
            "<b>Target:</b> %{y}%<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis=dict(range=[0, 110], title="Utilization %"),
        xaxis=dict(
            title="Machine"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemclick=False,
            itemdoubleclick=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div style="
            background:#1e2130;
            border-radius:10px;
            padding:14px 16px;
            margin-top:12px;
            text-align:center;
            border-left:4px solid #f1c40f;
        ">
            <div style="color:#ffffff; font-size:14px; font-weight:600; margin-bottom:6px;">
                Machine Utilization Targets
            </div>
            <div style="color:#cccccc; font-size:13px; letter-spacing:0.5px;">
                CVD-01 <b>86%</b> &nbsp;|&nbsp;
                CMP-01 <b>95%</b> &nbsp;|&nbsp;
                ETCH-01 <b>96%</b> &nbsp;|&nbsp;
                DIFF-01 <b>85%</b> &nbsp;|&nbsp;
                LITHO-01 <b>88%</b> &nbsp;|&nbsp;
                IMP-01 <b>89%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ── MACHINE PERFORMANCE TIMELINE ──────────────────────────────────────────
    st.markdown("#### Machine Performance Timeline")
    st.write("Chronological status timeline per machine · hover for details")

    def machine_util_pct(m):
        mdf = ov_df[ov_df["Machine_ID"] == m]
        tot = mdf["Duration_Min"].sum()
        up = mdf[mdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()
        return round((up / tot) * 100) if tot > 0 else 0

    machines = sorted(ov_df["Machine_ID"].unique(), key=machine_util_pct)

    for machine in machines:
        mdf = ov_df[ov_df["Machine_ID"] == machine]
        m_total = mdf["Duration_Min"].sum()
        m_up = mdf[mdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()
        m_util = round((m_up / m_total) * 100) if m_total > 0 else 0
        m_repair = int(mdf[mdf["Status"].isin(["WAIT_REPAIR", "IN_REPAIR"])]["Duration_Min"].sum())
        m_idle = int(mdf[mdf["Status"] == "IDLE"]["Duration_Min"].sum())
        m_pm = int(mdf[mdf["Status"].isin(["WAIT_PM", "IN_PM"])]["Duration_Min"].sum())
        util_color = get_util_color(m_util, st.session_state.util_threshold)


    tl_df = ov_df.copy()
    tl_df = tl_df.dropna(subset=["Start_Time", "End_Time"])
    tl_df = tl_df.sort_values("Start_Time")

    if tl_df.empty:
        st.warning("No timeline data available for this shift.")
    else:
        t_min = tl_df["Start_Time"].min()
        t_max = tl_df["End_Time"].max()
        total_span = (t_max - t_min).total_seconds() / 60.0

        if total_span <= 0:
            st.warning("Timeline span is zero — check Start_Time/End_Time columns.")
        else:
            # Build time axis labels (every 2 hours)
            import math
            axis_marks = []
            cur = t_min.replace(minute=0, second=0, microsecond=0)
            if cur < t_min:
                cur += pd.Timedelta(hours=1)
            while cur <= t_max:
                pct = ((cur - t_min).total_seconds() / 60.0) / total_span * 100
                axis_marks.append((pct, cur.strftime("%H:%M")))
                cur += pd.Timedelta(hours=2)

            axis_ticks_html = '<div style="position:relative; height:20px; margin-bottom:4px; margin-left:90px;">'
            for pct, label in axis_marks:
                axis_ticks_html += f'<span style="position:absolute; left:{pct:.1f}%; transform:translateX(-50%); font-size:10px; color:#888;">{label}</span>'
            axis_ticks_html += '</div>'
            st.markdown(axis_ticks_html, unsafe_allow_html=True)

            for machine in machines:
                mdf = tl_df[tl_df["Machine_ID"] == machine].sort_values("Start_Time")
                segments_html = ""
                for _, row in mdf.iterrows():
                    seg_start = (row["Start_Time"] - t_min).total_seconds() / 60.0
                    seg_dur = row["Duration_Min"]
                    left_pct = (seg_start / total_span) * 100
                    width_pct = (seg_dur / total_span) * 100
                    color = STATUS_COLORS.get(row["Status"], "#555555")
                    start_str = row["Start_Time"].strftime("%H:%M")
                    end_str = row["End_Time"].strftime("%H:%M") if pd.notna(row["End_Time"]) else "?"
                    reason = row.get("Downtime_Reason", "")
                    reason_str = f" · {reason}" if pd.notna(reason) and str(reason).strip() else ""
                    tooltip = f"{row['Status']}{reason_str} | {now} {start_str}–{end_str} ({int(seg_dur)} min)"
                    segments_html += f'<div title="{tooltip}" style="position:absolute; left:{left_pct:.2f}%; width:{max(width_pct, 0.3):.2f}%; background:{color}; height:100%; border-right:1px solid #000000; box-sizing:border-box;"></div>'

                st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:6px;">
                        <div style="width:85px; min-width:85px; font-size:12px; color:#cccccc; font-weight:bold; padding-right:8px; text-align:right;">{machine}</div>
                        <div style="flex:1; position:relative; height:22px; background:#2c2f3e; border-radius:4px; overflow:hidden;">
                            {segments_html}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Legend (reuse same STATUS_COLORS)
            tl_legend_html = '<div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; margin-left:90px;">'
            for status, color in STATUS_COLORS.items():
                tl_legend_html += f'<span style="font-size:12px; color:#aaaaaa;"><span style="display:inline-block; width:12px; height:12px; background:{color}; border-radius:2px; margin-right:4px;"></span>{status}</span>'
            tl_legend_html += '</div>'
            st.markdown(tl_legend_html, unsafe_allow_html=True)

    # ── AI SUMMARY SECTION ───────────────────────────────────────────────────
    if not ov_df.empty:
        if selected_machine == "All":
            summary_key = f"overview_{ov_active}_all"

            render_ai_summary_section(
                summary_key,
                build_ai_summary_prompt,
                ov_df,
                ov_shift_label,
                "All"
            )

            render_actions_next_shift_section(
                summary_key,
                build_next_actions_prompt,
                ov_df,
                ov_shift_label,
                "All"
            )

        else:
            summary_key = f"overview_{ov_active}_{selected_machine}"

            render_ai_summary_section(
                summary_key,
                build_ai_summary_prompt,
                ov_df,
                ov_shift_label,
                selected_machine
            )

            render_actions_next_shift_section(
                summary_key,
                build_next_actions_prompt,
                ov_df,
                ov_shift_label,
                selected_machine
            )
    else:
        st.warning("No data available for AI summary.")

    st.divider()

    st.markdown("#### ⚡ Clear Cache")

    if st.button("🧹 Clear AI Cache"):
        st.session_state.ai_summary_cache = {}
        st.session_state.actions_summary_cache = {}
        st.session_state.ai_summary = None
        st.session_state.actions_summary = None
        st.success("AI cache cleared.")

    st.divider()

    # ── DATASET PREVIEW SECTION ──────────────────────────────────────────────
    st.markdown("#### 📊 Dataset Preview")

    if st.button(
        "Show Dataset" if not st.session_state.show_dataset else "Hide Dataset",
        key="toggle_dataset_btn"
    ):
        st.session_state.show_dataset = not st.session_state.show_dataset
        st.rerun()

    if st.session_state.show_dataset:
        st.caption("Showing dataset based on current Shift and Machine filters.")
        st.dataframe(
            ov_df,
            use_container_width=True,
            hide_index=True
    )
    else:
        st.info("Dataset preview is hidden. Click **Show Dataset** to view it.")

# ── PAGE: UPLOADER ────────────────────────────────────────────────────────────
elif st.session_state.page == "upload":
    st.title("Upload Excel or CSV File")
    st.markdown("Upload your Excel or CSV file below to get started.")

    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=["xlsx", "xls", "csv"]
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")

        os.makedirs(DATASET_FOLDER, exist_ok=True)
        save_path = os.path.join(DATASET_FOLDER, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"💾 Saved to: `{save_path}`")

        try:
            uploaded_file.seek(0)
            df = read_uploaded_dataset(uploaded_file)

            st.session_state.df = df
            st.session_state.filename = uploaded_file.name
            st.session_state.page = "overview"
            st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Please upload an Excel or CSV file to continue.")

#testing
