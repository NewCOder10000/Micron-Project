import streamlit as st
import pandas as pd
import os
import google.genai as genai
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import requests

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
if "selected_shift" not in st.session_state:
    st.session_state.selected_shift = "total"
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None
if "ai_summary_key" not in st.session_state:
    st.session_state.ai_summary_key = None
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

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_MODEL = st.secrets["GEMINI_MODEL"]

@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"timeout": 30},
    )

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def metric_box(title, value, desc):
    st.markdown(f"""
        <div style="
            background-color: #1e2130;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 5px solid #4a9eff;
        ">
            <div style="font-size: 13px; color: #aaaaaa; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{value}</div>
            <div style="font-size: 11px; color: #777777; margin-top: 4px;">{desc}</div>
        </div>
    """, unsafe_allow_html=True)

def colored_metric(title, value, desc, color):
    st.markdown(f"""
        <div style="
            background-color: #1e2130;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 5px solid #4a9eff;
        ">
            <div style="font-size: 13px; color: #aaaaaa; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 28px; font-weight: bold; color: {color};">{value}</div>
            <div style="font-size: 11px; color: #777777; margin-top: 4px;">{desc}</div>
        </div>
    """, unsafe_allow_html=True)

def html_metric(title, value_html, desc):
    st.markdown(f"""
        <div style="
            background-color: #1e2130;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 5px solid #4a9eff;
        ">
            <div style="font-size: 13px; color: #aaaaaa; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 28px; font-weight: bold;">{value_html}</div>
            <div style="font-size: 11px; color: #777777; margin-top: 4px;">{desc}</div>
        </div>
    """, unsafe_allow_html=True)

def get_pct_color(pct):
    if pct >= 80:
        return "#2ecc71"
    elif pct >= 50:
        return "#f1c40f"
    else:
        return "#e74c3c"

def get_util_color(pct, threshold):
    return "#2ecc71" if pct >= threshold else "#e74c3c"

def call_gemini(prompt):
    import requests
    import time
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=(10, 60),
            )
            if resp.status_code in (429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.ConnectTimeout:
            if attempt == 2:
                return "❌ Connection timed out. Check your network or firewall."
            time.sleep(5)
        except requests.exceptions.ReadTimeout:
            if attempt == 2:
                return "❌ Gemini took too long to respond. Try again."
            time.sleep(5)
        except Exception as e:
            if attempt == 2:
                return f"❌ Error contacting Gemini: {e}"
            time.sleep(5)
    return "❌ Failed after 3 attempts."

def build_prompt_all(filtered_df, shift_label):
    fdf = filtered_df.copy()
    fdf["Status"] = fdf["Status"].str.strip()
    fdf["Machine_ID"] = fdf["Machine_ID"].str.strip()

    total_duration = fdf["Duration_Min"].sum()

    # IN_REPAIR alarm / fault frequency
    in_repair_df = fdf[fdf["Status"] == "IN_REPAIR"]
    alarm_series = in_repair_df["Downtime_Reason"].dropna().astype(str).str.strip()
    alarm_series = alarm_series[alarm_series != ""]
    top_alarms = alarm_series.value_counts().head(5)
    top_alarms_str = ", ".join(
        [f"{reason} ({count} occurrences)" for reason, count in top_alarms.items()]
    ) if not top_alarms.empty else "No IN_REPAIR fault types recorded"

    # Repair duration comparison
    in_repair_min = int(fdf[fdf["Status"] == "IN_REPAIR"]["Duration_Min"].sum())
    wait_repair_min = int(fdf[fdf["Status"] == "WAIT_REPAIR"]["Duration_Min"].sum())

    # WAIT_PM and WAIT_REPAIR percentage
    wait_pm_min = int(fdf[fdf["Status"] == "WAIT_PM"]["Duration_Min"].sum())
    wait_repair_pct = round((wait_repair_min / total_duration) * 100, 1) if total_duration > 0 else 0
    wait_pm_pct = round((wait_pm_min / total_duration) * 100, 1) if total_duration > 0 else 0

    # Non-machine failures
    non_machine_df = fdf[fdf["Status"].isin(["WAIT_REPAIR", "WAIT_PM"])]
    non_machine_reasons = non_machine_df["Downtime_Reason"].dropna().astype(str).str.strip()
    non_machine_reasons = non_machine_reasons[non_machine_reasons != ""]
    top_non_machine = non_machine_reasons.value_counts().head(5)
    top_non_machine_str = ", ".join(
        [f"{reason} ({count} occurrences)" for reason, count in top_non_machine.items()]
    ) if not top_non_machine.empty else "No people or parts delay reasons recorded"

    # Top 20% downtime contributors
    downtime_df = fdf[fdf["Status"] != "UP_PRODUCT"]
    machine_downtime = downtime_df.groupby("Machine_ID")["Duration_Min"].sum().sort_values(ascending=False)
    top_20_count = max(1, round(len(machine_downtime) * 0.2)) if len(machine_downtime) > 0 else 0
    top_20 = machine_downtime.head(top_20_count)
    top_20_str = ", ".join(
        [f"{machine} ({int(duration)} min)" for machine, duration in top_20.items()]
    ) if not top_20.empty else "No downtime contributors recorded"

    # Spare parts shortage flags
    spare_keywords = ["part", "parts", "spare", "material", "component", "stock"]
    spare_df = downtime_df[
        downtime_df["Downtime_Reason"].fillna("").astype(str).str.lower().str.contains(
            "|".join(spare_keywords)
        )
    ]
    spare_flags = spare_df.groupby("Machine_ID")["Duration_Min"].sum().sort_values(ascending=False)
    spare_flags_str = ", ".join(
        [f"{machine} ({int(duration)} min)" for machine, duration in spare_flags.items()]
    ) if not spare_flags.empty else "No spare parts shortage signals found"

    prompt = f"""
Generate structured shift-summary bullet points covering 7 insight areas:
1. Alarm frequency, focusing on the highest-occurrence IN_REPAIR fault types.
2. IN_REPAIR vs WAIT_REPAIR duration analysis.
3. Percentage of time in WAIT_PM and WAIT_REPAIR per shift.
4. Manpower shortage signals from WAIT_REPAIR and WAIT_PM dominance.
5. Non-machine failures, including people and parts delays.
6. Top 20% downtime contributors.
7. Spare parts shortage flags per machine.

Shift: {shift_label}

Available data:
- Top IN_REPAIR fault types: {top_alarms_str}
- IN_REPAIR duration: {in_repair_min} minutes
- WAIT_REPAIR duration: {wait_repair_min} minutes
- WAIT_PM duration: {wait_pm_min} minutes
- WAIT_REPAIR percentage of shift: {wait_repair_pct}%
- WAIT_PM percentage of shift: {wait_pm_pct}%
- Top non-machine delay reasons: {top_non_machine_str}
- Top 20% downtime contributors: {top_20_str}
- Spare parts shortage flags: {spare_flags_str}

Write exactly 7 bullet points using • as the bullet symbol.
Each bullet point should correspond to one insight area.
Be factual, direct, and operational.
Do not add headers.
Do not add closing remarks.
Do not use motivational language.
"""
    return prompt


def build_prompt_machine(machine_id, filtered_df, shift_label):
    fdf = filtered_df.copy()
    fdf["Status"] = fdf["Status"].str.strip()
    fdf["Machine_ID"] = fdf["Machine_ID"].str.strip()

    mdf = fdf[fdf["Machine_ID"] == machine_id].copy()
    total_duration = mdf["Duration_Min"].sum()

    # IN_REPAIR alarm / fault frequency
    in_repair_df = mdf[mdf["Status"] == "IN_REPAIR"]
    alarm_series = in_repair_df["Downtime_Reason"].dropna().astype(str).str.strip()
    alarm_series = alarm_series[alarm_series != ""]
    top_alarms = alarm_series.value_counts().head(5)
    top_alarms_str = ", ".join(
        [f"{reason} ({count} occurrences)" for reason, count in top_alarms.items()]
    ) if not top_alarms.empty else "No IN_REPAIR fault types recorded"

    # Repair duration comparison
    in_repair_min = int(mdf[mdf["Status"] == "IN_REPAIR"]["Duration_Min"].sum())
    wait_repair_min = int(mdf[mdf["Status"] == "WAIT_REPAIR"]["Duration_Min"].sum())

    # WAIT_PM and WAIT_REPAIR percentage
    wait_pm_min = int(mdf[mdf["Status"] == "WAIT_PM"]["Duration_Min"].sum())
    wait_repair_pct = round((wait_repair_min / total_duration) * 100, 1) if total_duration > 0 else 0
    wait_pm_pct = round((wait_pm_min / total_duration) * 100, 1) if total_duration > 0 else 0

    # Non-machine failures
    non_machine_df = mdf[mdf["Status"].isin(["WAIT_REPAIR", "WAIT_PM"])]
    non_machine_reasons = non_machine_df["Downtime_Reason"].dropna().astype(str).str.strip()
    non_machine_reasons = non_machine_reasons[non_machine_reasons != ""]
    top_non_machine = non_machine_reasons.value_counts().head(5)
    top_non_machine_str = ", ".join(
        [f"{reason} ({count} occurrences)" for reason, count in top_non_machine.items()]
    ) if not top_non_machine.empty else "No people or parts delay reasons recorded"

    # Machine downtime contribution
    machine_downtime = int(mdf[mdf["Status"] != "UP_PRODUCT"]["Duration_Min"].sum())

    # Spare parts shortage flags
    spare_keywords = ["part", "parts", "spare", "material", "component", "stock"]
    spare_df = mdf[
        mdf["Downtime_Reason"].fillna("").astype(str).str.lower().str.contains(
            "|".join(spare_keywords)
        )
    ]
    spare_min = int(spare_df["Duration_Min"].sum())
    spare_reasons = spare_df["Downtime_Reason"].dropna().astype(str).str.strip()
    spare_reasons = spare_reasons[spare_reasons != ""]
    spare_reasons_str = ", ".join(spare_reasons.value_counts().head(3).index.tolist()) \
        if not spare_reasons.empty else "No spare parts shortage signals found"

    prompt = f"""
Generate structured machine shift-summary bullet points covering 7 insight areas:
1. Alarm frequency, focusing on the highest-occurrence IN_REPAIR fault types.
2. IN_REPAIR vs WAIT_REPAIR duration analysis.
3. Percentage of time in WAIT_PM and WAIT_REPAIR for this shift.
4. Manpower shortage signals from WAIT_REPAIR and WAIT_PM dominance.
5. Non-machine failures, including people and parts delays.
6. Downtime contribution of this machine.
7. Spare parts shortage flags for this machine.

Machine: {machine_id}
Shift: {shift_label}

Available data:
- Top IN_REPAIR fault types: {top_alarms_str}
- IN_REPAIR duration: {in_repair_min} minutes
- WAIT_REPAIR duration: {wait_repair_min} minutes
- WAIT_PM duration: {wait_pm_min} minutes
- WAIT_REPAIR percentage of shift: {wait_repair_pct}%
- WAIT_PM percentage of shift: {wait_pm_pct}%
- Top non-machine delay reasons: {top_non_machine_str}
- Total downtime contribution: {machine_downtime} minutes
- Spare parts shortage duration: {spare_min} minutes
- Spare parts shortage reasons: {spare_reasons_str}

Write exactly 7 bullet points using • as the bullet symbol.
Each bullet point should correspond to one insight area.
Be factual, direct, and operational.
Do not add headers.
Do not add closing remarks.
Do not use motivational language.
"""
    return prompt

def render_ai_summary_section(summary_key, prompt_fn, *prompt_args):
    st.divider()
    st.markdown("### 🤖 AI Summary")

    if st.button("✨ Generate AI Summary", key=f"ai_btn_{summary_key}"):
        with st.spinner("Generating summary with Gemini..."):
            prompt = prompt_fn(*prompt_args)
            result = call_gemini(prompt)
            st.session_state.ai_summary = result
            st.session_state.ai_summary_key = summary_key

    if st.session_state.ai_summary and st.session_state.ai_summary_key == summary_key:
        lines = [
            l.strip()
            for l in st.session_state.ai_summary.split("\n")
            if l.strip().startswith("•")
        ]

        if not lines:
            lines = [
                "• " + l.strip()
                for l in st.session_state.ai_summary.split("•")
                if l.strip()
            ]

        if lines:
            bullet_html = "<br>".join([
                f'<div style="margin-bottom:8px;">{line}</div>'
                for line in lines[:7]
            ])

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

        else:
            st.markdown(f"""
                <div style="
                    background:#1e2130;
                    border-radius:10px;
                    padding:16px 20px;
                    margin-top:8px;
                    border-left:5px solid #4a9eff;
                ">
                    <span style="color:#e0e0e0; font-size:14px; line-height:1.6;">
                        {st.session_state.ai_summary}
                    </span>
                </div>
            """, unsafe_allow_html=True)

# ── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────────
if st.session_state.df is not None and st.session_state.page != "upload":
    with st.sidebar:
        st.markdown("## 🗂️ Navigation")
        if st.button("📋 Overview", use_container_width=True,
                     type="primary" if st.session_state.page == "overview" else "secondary"):
            st.session_state.page = "overview"
            st.rerun()
        st.divider()

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

    st.markdown(f"#### Fleet Summary · {ov_shift_label}")
    # ── Additional Fleet Metrics ────────────────────────────────────────────────
    machines_running = ov_df[ov_df["Status"] == "UP_PRODUCT"]["Machine_ID"].nunique()
    total_downtime = int(ov_df[ov_df["Status"] != "UP_PRODUCT"]["Duration_Min"].sum())

    # Row 1: 4 boxes
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        colored_metric(
            "Fleet Utilization",
            f"{fleet_util}%",
            f"UP_PRODUCT across {total_machines} machines",
            get_util_color(fleet_util, st.session_state.util_threshold)
        )

    with fc2:
        if selected_machine != "All":
            # For individual machine view
            machine_status = "Active" if fleet_util > 0 else "Inactive"
            machine_status_color = "#2ecc71" if fleet_util > 0 else "#e74c3c"

            html_metric(
                "Machine Status",
                f'<span style="color:{machine_status_color};">{machine_status}</span>',
                f"Utilization: {fleet_util}%"
            )

        else:
            # For All machines view
            running_pct = (machines_running / total_machines) * 100 if total_machines > 0 else 0
            running_color = "#2ecc71" if running_pct >= 50 else "#e74c3c"

            html_metric(
                "Machines Running",
                f'<span style="color:{running_color};">{machines_running}</span> '
                f'<span style="color:#ffffff;">out of {total_machines}</span>',
                "Machines with UP_PRODUCT activity"
            )

    with fc3:
        metric_box(
            "Total Downtime",
            f"{total_downtime} min",
            "All non-UP_PRODUCT time"
        )

    with fc4:
        metric_box(
            "Total Repair Time",
            f"{fleet_repair} min",
            "WAIT_REPAIR + IN_REPAIR"
        )

    st.write("")

    # Row 2: 4 boxes
    fc5, fc6, fc7, fc8 = st.columns(4)

    with fc5:
        metric_box(
            "Total Idle Time",
            f"{fleet_idle} min",
            "IDLE status across all machines"
        )

    with fc6:
        metric_box(
            "Total PM Time",
            f"{fleet_pm} min",
            "WAIT_PM + IN_PM"
        )

    st.divider()

    # ── UTILISATION THRESHOLD ON OVERVIEW PAGE ───────────────────────────────
    st.divider()
    st.markdown("#### 🎯 Utilisation Threshold")

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

    th_col1, th_col2 = st.columns([2, 1])

    with th_col1:
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

    with th_col2:
        st.markdown(
            f"""
            <div style="
                background-color:#1e2130;
                border-radius:12px;
                padding:22px 24px;
                margin-top:28px;
                border-left:5px solid #4a9eff;
                min-height:92px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                gap:10px;
                justify-content:center;
                align-items:center;
            ">
                <div style="color:#2ecc71; font-weight:bold; font-size:15px;">
                    🟢 ≥ {st.session_state['util_threshold']}%
                </div>
                <div style="color:#e74c3c; font-weight:bold; font-size:15px;">
                    🔴 &lt; {st.session_state['util_threshold']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── MACHINE PERFORMANCE TIMELINE ──────────────────────────────────────────
    st.divider()
    st.markdown("#### Machine Performance Timeline")
    st.write("Chronological status timeline per machine · hover for details")

    STATUS_COLORS = {
        "UP_PRODUCT": "#008000",
        "IDLE":       "#5F5FFF",
        "WAIT_REPAIR":"#FF0000",
        "IN_REPAIR":  "#FF0000",
        "WAIT_PM":    "#FFC0CB",
        "IN_PM":      "#FFC0CB",
    }

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

    # ── COLORS ────────────────────────────────────────────────────
    colors = [
        "#2ecc71" if u >= t else "#e74c3c"
        for u, t in zip(chart_df["Utilization"], chart_df["Target"])
    ]

# ── PLOT ──────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=chart_df["Machine_ID"],
        y=chart_df["Utilization"],
        marker_color=colors,
        text=[f"{u}%" for u in chart_df["Utilization"]],
        textposition="outside"
    ))

    st.markdown("####  Machine Utilization vs Target")

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis=dict(range=[0, 110]),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div style='color:#aaa; font-size:12px; text-align:center; margin-top:10px;'>"
        "Targets: CMP-01 95% | CVD-01 86% | DIFF-01 85% | ETCH-01 96% | IMP-01 89% | LITHO-01 88%"
        "</div>",
        unsafe_allow_html=True
    )

    # ── DATASET PREVIEW SECTION ──────────────────────────────────────────────
    st.divider()
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

    # ── AI SUMMARY SECTION ───────────────────────────────────────────────────
    if not ov_df.empty:
        if selected_machine == "All":
            summary_key = f"overview_{ov_active}_all"
            render_ai_summary_section(
                summary_key,
                build_prompt_all,
                ov_df,
                ov_shift_label
            )
        else:
            summary_key = f"overview_{ov_active}_{selected_machine}"
            render_ai_summary_section(
                summary_key,
                build_prompt_machine,
                selected_machine,
                ov_df,
                ov_shift_label
            )
    else:
        st.warning("No data available for AI summary.")

# ── PAGE: UPLOADER ────────────────────────────────────────────────────────────
elif st.session_state.page == "upload":
    st.title("Upload Excel File")
    st.markdown("Upload your Excel file below to get started.")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")

        os.makedirs(DATASET_FOLDER, exist_ok=True)
        save_path = os.path.join(DATASET_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"💾 Saved to: `{save_path}`")

        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.filename = uploaded_file.name
            st.session_state.page = "overview"
            st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Please upload an Excel file to continue.")

#lol