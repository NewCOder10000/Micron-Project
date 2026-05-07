import streamlit as st
import pandas as pd
import os
import google.genai as genai
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

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

    machine_util = {}
    for mid, mdf in fdf.groupby("Machine_ID"):
        tot = mdf["Duration_Min"].sum()
        up = mdf[mdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()
        machine_util[mid] = round((up / tot) * 100) if tot > 0 else 0
    worst_machine = min(machine_util, key=machine_util.get)
    worst_util = machine_util[worst_machine]
    worst_downtime = int(fdf[(fdf["Machine_ID"] == worst_machine) & (fdf["Status"] != "UP_PRODUCT")]["Duration_Min"].sum())

    downtime_df = fdf[fdf["Status"] != "UP_PRODUCT"]
    reason_series = downtime_df["Downtime_Reason"].dropna().str.strip()
    reason_series = reason_series[reason_series != ""]
    top3 = reason_series.value_counts().head(3)
    top3_str = ", ".join([f"{r} ({c} occurrences)" for r, c in top3.items()]) if not top3.empty else "No downtime reasons recorded"

    prompt = f"""You are a shift supervisor writing a concise handover note. Be factual and direct — no motivational phrases, no filler sentences, no conclusions like "let's get it back up" or "our process needs fine-tuning". Just clear, useful observations and actions.

Shift: {shift_label}
Worst performing machine: {worst_machine} - only {worst_util}% utilization, lost {worst_downtime} minutes to downtime
Top 3 downtime reasons across all machines: {top3_str}

Write exactly 3 bullet points using • as the bullet symbol. No headers, no extra text, just the 3 bullets.

• Bullet 1: State which machine struggled most, its utilization %, and how many minutes were lost. One sentence, factual only.
• Bullet 2: Explain what the top 3 downtime reasons reveal about what's going wrong on the floor. One to two sentences, no filler.
• Bullet 3: Give 1-2 specific, actionable things the next shift should do based on the downtime patterns. No motivational language.

IMPORTANT: Do not add any closing remarks, encouragement, or commentary beyond the 3 bullets."""
    return prompt


def build_prompt_machine(machine_id, filtered_df, shift_label):
    fdf = filtered_df.copy()
    fdf["Status"] = fdf["Status"].str.strip()
    mdf = fdf[fdf["Machine_ID"].str.strip() == machine_id]

    tot = mdf["Duration_Min"].sum()
    up = mdf[mdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum()
    util = round((up / tot) * 100) if tot > 0 else 0
    downtime_min = int(tot - up)

    downtime_df = mdf[mdf["Status"] != "UP_PRODUCT"]
    reason_series = downtime_df["Downtime_Reason"].dropna().str.strip()
    reason_series = reason_series[reason_series != ""]
    top3 = reason_series.value_counts().head(3)
    top3_str = ", ".join([f"{r} ({c} occurrences)" for r, c in top3.items()]) if not top3.empty else "No downtime reasons recorded"

    prompt = f"""You are a shift supervisor writing a concise machine handover note. Be factual and direct — no motivational phrases, no filler sentences, no conclusions like "let's improve this" or "we need to fine-tune our process". Just clear, useful observations and actions.

Machine: {machine_id}
Shift: {shift_label}
Utilization: {util}%
Total downtime: {downtime_min} minutes
Top 3 downtime reasons: {top3_str}

Write exactly 3 bullet points using • as the bullet symbol. No headers, no extra text, just the 3 bullets.

• Bullet 1: State the machine's utilization and downtime minutes. One sentence, factual only.
• Bullet 2: Explain what the top downtime reasons suggest is happening with this machine. One to two sentences, no filler.
• Bullet 3: Give 1-2 specific, actionable things the next operator should do for this machine. No motivational language.

IMPORTANT: Do not add any closing remarks, encouragement, or commentary beyond the 3 bullets."""
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
        lines = [l.strip() for l in st.session_state.ai_summary.split("\n") if l.strip().startswith("•")]
        if not lines:
            lines = ["• " + l.strip() for l in st.session_state.ai_summary.split("•") if l.strip()]
        if lines:
            for line in lines[:3]:
                st.markdown(f"""
                    <div style="background:#1e2130; border-radius:8px; padding:12px 16px; margin-bottom:8px; border-left:4px solid #4a9eff;">
                        <span style="color:#e0e0e0; font-size:14px;">{line}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background:#1e2130; border-radius:8px; padding:12px 16px; border-left:4px solid #4a9eff;">
                    <span style="color:#e0e0e0; font-size:14px;">{st.session_state.ai_summary}</span>
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
    st.title(f"📋 Overview - {st.session_state.filename}")

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

    with fc7:
        metric_box(
            "Placeholder",
            "-",
            "Reserved for future metric"
        )

    with fc8:
        metric_box(
            "Placeholder",
            "-",
            "Reserved for future metric"
        )

    st.divider()

    # ── UTILISATION THRESHOLD ON OVERVIEW PAGE ───────────────────────────────
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

    th_col1, th_col2, th_col3 = st.columns([3, 1, 2])

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

    with th_col2:
        st.number_input(
            "Manual input",
            min_value=0,
            max_value=100,
            step=1,
            value=int(st.session_state["_thresh_input"]),
            key="_thresh_input",
            on_change=_sync_from_input,
        )

    with th_col3:
        st.markdown(
            f"""
            <div style="
                background-color:#1e2130;
                border-radius:10px;
                padding:14px 18px;
                margin-top:24px;
                border-left:5px solid #4a9eff;
            ">
                <span style="color:#2ecc71; font-weight:bold;">🟢 ≥ {st.session_state['util_threshold']}%</span>
                <br>
                <span style="color:#e74c3c; font-weight:bold;">🔴 &lt; {st.session_state['util_threshold']}%</span>
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

    ### Stupid bar graph
    st.markdown("#### Machine Performance")
    st.write("PLEASE SHUT UP I GO DIE!")

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

    fig.update_layout(
        title="Machine Utilization vs Target",
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

# ── PAGE: DATA VIEWER ──────────────────────────────────────────────────────────
elif st.session_state.page == "viewer":
    df = st.session_state.df

    with st.sidebar:
        st.header("🏭 Filter")
        machine_ids = ["All"] + sorted(df["Machine_ID"].str.strip().unique().tolist())
        selected_machine = st.selectbox("Machine", options=machine_ids, index=0, label_visibility="visible")

    st.title("FabSight - Shift Performance Dashboard")
    st.write("By Uptime Guardians")

    active = st.session_state.selected_shift
    st.markdown(f"""
        <style>
        div[data-testid="column"]:nth-of-type(2) button {{
            {"font-weight: bold; border: 2px solid #ffffff;" if active == "total" else ""}
        }}
        div[data-testid="column"]:nth-of-type(3) button {{
            {"font-weight: bold; border: 2px solid #ffffff;" if active == "day" else ""}
        }}
        div[data-testid="column"]:nth-of-type(4) button {{
            {"font-weight: bold; border: 2px solid #ffffff;" if active == "night" else ""}
        }}
        </style>
    """, unsafe_allow_html=True)

    _, shift_col1, shift_col2, shift_col3, _ = st.columns([1.5, 1, 1, 1, 1.5])

    with shift_col1:
        label = "🕐 Total (1440 min)" + (" ✔" if active == "total" else "")
        if st.button(label, key="shift_btn_total", use_container_width=True):
            st.session_state.selected_shift = "total"
            st.rerun()

    with shift_col2:
        label = "☀️ Day (720 min)" + (" ✔" if active == "day" else "")
        if st.button(label, key="shift_btn_day", use_container_width=True):
            st.session_state.selected_shift = "day"
            st.rerun()

    with shift_col3:
        label = "🌙 Night (720 min)" + (" ✔" if active == "night" else "")
        if st.button(label, key="shift_btn_night", use_container_width=True):
            st.session_state.selected_shift = "night"
            st.rerun()

    st.write("")

    if selected_machine == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["Machine_ID"].str.strip() == selected_machine].copy()

    TIME_COL = "Start_Time"
    if TIME_COL in filtered_df.columns:
        filtered_df[TIME_COL] = pd.to_datetime(filtered_df[TIME_COL], errors="coerce")
        hour = filtered_df[TIME_COL].dt.hour
        if st.session_state.selected_shift == "day":
            filtered_df = filtered_df[(hour >= 7) & (hour < 19)]
        elif st.session_state.selected_shift == "night":
            filtered_df = filtered_df[(hour < 7) | (hour >= 19)]

    shift_label = {
        "total": "Total Shift (Day & Night)",
        "day":   "Day Shift (07:00 – 19:00)",
        "night": "Night Shift (19:00 – 07:00)",
    }[st.session_state.selected_shift]

    status_col = df["Status"].str.strip()
    filtered_status = filtered_df["Status"].str.strip()

    if selected_machine == "All":
        total_machines = df["Machine_ID"].str.strip().nunique()
        running_machines = filtered_df[filtered_status == "UP_PRODUCT"]["Machine_ID"].str.strip().nunique()
        box1_title = "Machines Running"
        box1_ratio = round((running_machines / total_machines) * 100) if total_machines > 0 else 0
        box1_color = get_util_color(box1_ratio, st.session_state.util_threshold)
        box1_desc = f"Currently producing · {shift_label}"
        box1_value_html = f'<span style="color:{box1_color};">{running_machines}</span><span style="color:#ffffff;"> of {total_machines}</span>'
    else:
        is_running = (filtered_status == "UP_PRODUCT").any()
        box1_title = "Machine Status"
        box1_color = "#2ecc71" if is_running else "#e74c3c"
        box1_desc = f"Based on UP_PRODUCT · {shift_label}"
        box1_value_html = f'<span style="color:{box1_color};">{"Running" if is_running else "Not Running"}</span>'

    total_duration = filtered_df["Duration_Min"].sum()
    up_duration = filtered_df[filtered_status == "UP_PRODUCT"]["Duration_Min"].sum()
    avg_util = round((up_duration / total_duration) * 100) if total_duration > 0 else 0
    box2_color = get_util_color(avg_util, st.session_state.util_threshold)

    # Box 3: Shift downtime = not UP_PRODUCT and not IDLE
    shift_downtime = int(filtered_df[~filtered_status.isin(["UP_PRODUCT", "IDLE"])]["Duration_Min"].sum())

    # Box 4: Fault repair time = WAIT_REPAIR + IN_REPAIR
    total_repair = int(filtered_df[filtered_status.isin(["WAIT_REPAIR", "IN_REPAIR"])]["Duration_Min"].sum())

    st.title(f"📊 {st.session_state.filename} - {shift_label}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        html_metric(box1_title, box1_value_html, box1_desc)
    with col2:
        colored_metric("Average Utilization", f"{avg_util}%", "Based on UP_PRODUCT status", box2_color)
    with col3:
        colored_metric("Total Shift Downtime", f"{shift_downtime} min", "WAIT_REPAIR, IN_REPAIR, WAIT_PM, IN_PM", "#ffffff")
    with col4:
        colored_metric("Total (Fault) Repair Time", f"{total_repair} min", "When status is WAIT_REPAIR or IN_REPAIR", "#ffffff")

    st.divider()
    # Optional: only show raw data when explicitly requested
    show_raw = st.sidebar.checkbox("Show raw data (dashboard)", value=False, help="Enable to view the filtered rows/columns on the dashboard")
    if show_raw:
        st.markdown(f"**Rows:** {filtered_df.shape[0]} | **Columns:** {filtered_df.shape[1]}")
        display_df = filtered_df.reset_index(drop=True)
        display_df.index += 1
        st.dataframe(display_df, use_container_width=True)

    # ── AI SUMMARY ────────────────────────────────────────────────────────────
    summary_key = f"{selected_machine}__{st.session_state.selected_shift}"
    if selected_machine == "All":
        render_ai_summary_section(summary_key, build_prompt_all, filtered_df, shift_label)
    else:
        render_ai_summary_section(summary_key, build_prompt_machine, selected_machine, filtered_df, shift_label)

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