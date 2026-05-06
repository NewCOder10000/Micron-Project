# ============================================================
#  FabSight – Shift Performance Dashboard  (No API / Free)
#
#  HOW TO RUN:
#    pip install streamlit plotly pandas openpyxl requests
#    streamlit run fabsight_simple.py
#
#  LAYOUT ORDER:
#    1. Title + byline
#    2. Date + 5-column KPI table
#    3. Colour legend (global)
#    4. ── separator ──
#    5. Shift selector buttons  [Day | Night | Compare]
#    6. Gantt card  (legend strip above timeline)
#    7. Summary + Priorities
#    8. Machine AI Advisor  (shift-comparison arrows)
#
#  STATUS COLOURS:
#    UP_PRODUCT  #4CAF50  green
#    IDLE        #2196F3  blue
#    IN_REPAIR   #FF0000  red
#    WAIT_REPAIR #FF6666  light-red
#    IN_PM       #FFC0CB  pink
#    WAIT_PM     #FFD9E0  light-pink
#
#  KPI DEFINITIONS:
#    Avg Utilisation   = UP_PRODUCT only
#    Total Downtime    = IN_REPAIR + WAIT_REPAIR + IN_PM + WAIT_PM
#    Fault Repair Time = IN_REPAIR + WAIT_REPAIR only
#    Total Idle Time   = IDLE only
# ============================================================

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
from datetime import date
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SHIFT_TOTAL_MIN = 720
MACHINES        = ["CMP-01", "CVD-01", "DIFF-01", "ETCH-01", "IMP-01", "LITHO-01"]
MACHINE_TYPE    = {
    "CMP-01": "CMP",  "CVD-01": "CVD",   "DIFF-01": "Diff",
    "ETCH-01": "Etch", "IMP-01": "Imp",  "LITHO-01": "Litho",
}

# Status colours — WAIT variants slightly lighter so white boundary line is visible
STATUS_COLOR = {
    "UP_PRODUCT":  "#4CAF50",
    "IDLE":        "#2196F3",
    "IN_REPAIR":   "#FF0000",
    "WAIT_REPAIR": "#FF6666",
    "IN_PM":       "#FFC0CB",
    "WAIT_PM":     "#FFD9E0",
}
STATUS_LABEL = {
    "UP_PRODUCT":  "Running",
    "IDLE":        "Idle",
    "IN_REPAIR":   "In Repair",
    "WAIT_REPAIR": "Wait Repair",
    "IN_PM":       "In PM",
    "WAIT_PM":     "Wait PM",
}

REPAIR_GROUP = {"IN_REPAIR", "WAIT_REPAIR"}
PM_GROUP     = {"IN_PM", "WAIT_PM"}

# Inline legend shown above each Gantt chart
GANTT_LEGEND_ENTRIES = [
    ("UP_PRODUCT",  "Running (UP_PRODUCT)"),
    ("IDLE",        "Idle"),
    ("IN_REPAIR",   "In Repair"),
    ("WAIT_REPAIR", "Wait Repair"),
    ("IN_PM",       "In PM"),
    ("WAIT_PM",     "Wait PM"),
]

# Button colours
BTN_DAY     = dict(bg="#b8860b", border_active="#FFD700", border_idle="#7a5c00")
BTN_NIGHT   = dict(bg="#1565C0", border_active="#2196F3", border_idle="#0d47a1")
BTN_COMPARE = dict(bg="#2E7D32", border_active="#4CAF50", border_idle="#1b5e20")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FabSight – Shift Performance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ── */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{
    background:#1e1e1e;color:#e2e8f0}
.block-container{padding:1.4rem 2rem}

/* ── header ── */
.fab-title  {font-size:28px;font-weight:900;color:#f1f5f9;margin-bottom:2px}
.fab-byline {font-style:italic;color:#9ca3af;font-size:13px;margin-bottom:18px}
.date-line  {font-size:15px;color:#e2e8f0;margin-bottom:22px}
.date-line u{text-decoration:underline}

/* ── KPI table ── */
.kpi-table{width:100%;border-collapse:collapse;margin-bottom:10px}
.kpi-table td{
    border:1px solid #4b5563;padding:16px 18px;
    font-size:15px;font-weight:500;color:#e2e8f0;
    background:#2a2a2a;vertical-align:top}
.kpi-header{font-size:16px;font-weight:600;color:#f1f5f9;
            text-align:center;line-height:1.3}
.kpi-val{font-size:28px;font-weight:800;text-align:center;margin-top:8px}
.c-green {color:#4CAF50}
.c-yellow{color:#FFC107}
.c-red   {color:#FF5252}
.c-blue  {color:#2196F3}
.colour-key{font-size:12px;color:#9ca3af;margin-top:10px}

/* ── section labels ── */
.section-label{font-size:10px;font-weight:700;letter-spacing:1.5px;
               color:#6b7280;text-transform:uppercase;margin:8px 0 4px 0}
.section-sub{font-size:10px;color:#4b5563;margin-bottom:12px}

/* ── shift pill ── */
.shift-pill{display:inline-block;padding:5px 14px;border-radius:20px;
            font-size:12px;font-weight:600;margin-bottom:14px}
.pill-day  {background:#1c2e1c;color:#4CAF50;border:1px solid #4CAF50}
.pill-night{background:#1c1e2e;color:#818cf8;border:1px solid #818cf8}

/* ── gantt legend strip ── */
.gantt-legend{display:flex;flex-wrap:wrap;gap:4px 14px;
              margin-bottom:10px;padding:6px 0}
.gantt-legend-item{display:inline-flex;align-items:center;
                   gap:5px;font-size:11px;color:#9ca3af}
.gantt-legend-swatch{width:11px;height:11px;border-radius:2px;
                     display:inline-block;flex-shrink:0}

/* ── machine card ── */
.mcard{background:#2a2a2a;border-radius:10px;
       padding:12px 14px 10px 14px;border:1px solid #374151;margin-bottom:10px}
.mcard-header{display:flex;justify-content:space-between;
              align-items:flex-start;margin-bottom:2px}
.mcard-id   {font-size:14px;font-weight:700;color:#f1f5f9}
.mcard-type {font-size:10px;color:#6b7280;margin-top:1px}
.mcard-badge{font-size:9px;font-weight:700;padding:2px 8px;
             border-radius:10px;background:#14532d;color:#4CAF50}
.mcard-bar  {height:5px;border-radius:3px;margin:7px 0 8px 0;
             display:flex;overflow:hidden;gap:1px}
.mcard-stats{font-size:11px;color:#9ca3af;margin-bottom:6px}
.mcard-stats b{color:#f1f5f9}
.concern-badge{display:inline-block;font-size:9px;font-weight:700;
               padding:2px 8px;border-radius:10px;margin-top:2px}
.cb-fault{background:#450a0a;color:#FF5252}
.cb-pm   {background:#3d1020;color:#FFC0CB}
.cb-idle {background:#0a1a2e;color:#2196F3}

/* ── summary / priority cards ── */
.summary-card{background:#2a2a2a;border-radius:10px;padding:14px 16px;
              border:1px solid #374151;margin-bottom:14px}
.summary-badge{display:inline-block;background:#1e3a5f;color:#93c5fd;
               font-size:9px;font-weight:700;padding:2px 8px;
               border-radius:4px;letter-spacing:1px;margin-bottom:8px}
.pri-item{display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;
          font-size:12px;color:#cbd5e1;line-height:1.45}
.dot-r{width:9px;height:9px;border-radius:50%;background:#FF0000;
       margin-top:3px;flex-shrink:0}
.dot-b{width:9px;height:9px;border-radius:50%;background:#2196F3;
       margin-top:3px;flex-shrink:0}
.dot-g{width:9px;height:9px;border-radius:50%;background:#4CAF50;
       margin-top:3px;flex-shrink:0}

/* ── AI advisor snapshot tile ── */
.snap-tile{background:#2a2a2a;border-radius:10px;padding:14px;
           border:1px solid #374151;text-align:center}
.snap-label{font-size:11px;color:#6b7280;margin-bottom:4px;
            text-transform:uppercase;letter-spacing:.8px}
.snap-value{font-size:22px;font-weight:800;color:#f1f5f9}

/* ── AI advisor output ── */
.ai-box{background:#1a2a1a;border:1px solid #4CAF50;border-radius:10px;
        padding:16px 18px;margin-top:8px}
.ai-box-title{color:#4CAF50;font-weight:700;font-size:13px;margin-bottom:10px}
.ai-action{background:#2a2a2a;border-left:3px solid #4CAF50;padding:10px 14px;
           border-radius:4px;margin-bottom:8px;font-size:13px;
           color:#e2e8f0;line-height:1.5}
.ai-action b{color:#4CAF50}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load and clean the shift Excel file."""
    df = pd.read_excel(file, sheet_name="shift_data")
    df = df[["Machine_ID", "Shift", "Start_Time", "End_Time",
             "Duration_Min", "Status", "Downtime_Reason"]].copy()
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["End_Time"]   = pd.to_datetime(df["End_Time"])
    df["Status"]     = df["Status"].str.strip()
    return df


def calc_metrics(df: pd.DataFrame, shift: str) -> dict:
    """
    Compute per-machine stats and shift-level aggregates.

    Returned keys:
        stats       – dict keyed by machine name
        avg_util    – average UP_PRODUCT utilisation %
        total_down  – fault + PM minutes (all machines)
        total_fault – IN_REPAIR + WAIT_REPAIR minutes
        total_idle  – IDLE minutes
    """
    sdf, total = df[df["Shift"] == shift], SHIFT_TOTAL_MIN
    stats = {}

    for m in MACHINES:
        mdf      = sdf[sdf["Machine_ID"] == m]
        run      = int(mdf[mdf["Status"] == "UP_PRODUCT"]["Duration_Min"].sum())
        fault    = int(mdf[mdf["Status"].isin(REPAIR_GROUP)]["Duration_Min"].sum())
        pm       = int(mdf[mdf["Status"].isin(PM_GROUP)]["Duration_Min"].sum())
        idle     = int(mdf[mdf["Status"] == "IDLE"]["Duration_Min"].sum())
        downtime = fault + pm
        reasons  = (mdf[mdf["Downtime_Reason"].notna()]["Downtime_Reason"]
                    .unique().tolist())

        if fault >= 60:  concern, cc = f"{fault*100//total}% fault", "cb-fault"
        elif pm  >= 90:  concern, cc = f"{pm*100//total}% PM",       "cb-pm"
        elif idle >= 60: concern, cc = f"{idle*100//total}% idle",   "cb-idle"
        else:            concern, cc = None, ""

        stats[m] = dict(
            run=run, fault=fault, pm=pm, idle=idle, downtime=downtime,
            util_pct     = run*100//total,
            fault_pct    = fault*100//total,
            pm_pct       = pm*100//total,
            idle_pct     = idle*100//total,
            downtime_pct = downtime*100//total,
            reasons=reasons, concern=concern, concern_cls=cc,
        )

    return dict(
        stats       = stats,
        avg_util    = sum(v["util_pct"] for v in stats.values()) // len(MACHINES),
        total_down  = sum(v["downtime"] for v in stats.values()),
        total_fault = sum(v["fault"]    for v in stats.values()),
        total_idle  = sum(v["idle"]     for v in stats.values()),
    )


def worst_to_best(metrics: dict) -> list[str]:
    """Machines sorted ascending by utilisation % (worst first)."""
    return sorted(metrics["stats"], key=lambda m: metrics["stats"][m]["util_pct"])


def compare_shifts(day_m: dict, night_m: dict, machine: str) -> dict[str, dict]:
    """
    For each stat field compare Day vs Night values.
    Returns {field: {day, night, pct_change}} where:
        pct_change = rounded % difference (Night vs Day), or None if day == 0.
    Fields: run, fault, pm, idle, downtime.
    """
    fields = ["run", "fault", "pm", "idle", "downtime"]
    dv, nv = day_m["stats"][machine], night_m["stats"][machine]
    result = {}
    for f in fields:
        d, n = dv[f], nv[f]
        if d > 0:
            pct = round((n - d) / d * 100, 1)
        elif n > 0:
            pct = 100.0   # day was 0, night is non-zero → +100%
        else:
            pct = None    # both zero — no change to report
        result[f] = dict(day=d, night=n, pct_change=pct)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  GANTT CHART
# ─────────────────────────────────────────────────────────────────────────────
def _to_chart_hour(clock_h: float, is_night: bool) -> float:
    """
    Map fractional clock-hour to linear x-axis.
    Night: post-midnight hours (00-11) → 24-35 so axis runs 19→31.
    """
    return clock_h + 24 if (is_night and clock_h < 12) else clock_h


def _add_boundary_lines(fig: go.Figure, sdf: pd.DataFrame,
                         machine_order: list, is_night: bool) -> None:
    """White vertical line at every WAIT→IN transition."""
    transitions = [("WAIT_REPAIR", "IN_REPAIR"), ("WAIT_PM", "IN_PM")]
    for machine in machine_order:
        mdf      = sdf[sdf["Machine_ID"] == machine].sort_values("Start_Time")
        statuses = mdf["Status"].tolist()
        for i in range(len(statuses) - 1):
            for from_s, to_s in transitions:
                if statuses[i] == from_s and statuses[i + 1] == to_s:
                    row   = mdf.iloc[i + 1]
                    x_pos = _to_chart_hour(
                        row["Start_Time"].hour + row["Start_Time"].minute / 60,
                        is_night,
                    )
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos], y=[machine, machine],
                        mode="lines",
                        line=dict(color="#FFFFFF", width=2),
                        showlegend=False, hoverinfo="skip",
                    ))


def build_timeline(df: pd.DataFrame, shift: str,
                   machine_order: list) -> go.Figure:
    """
    Horizontal Gantt chart.
    machine_order should be worst→best; the chart reverses it so
    the worst machine appears at the TOP (Plotly renders bottom-up).
    """
    sdf           = df[df["Shift"] == shift].copy()
    is_night      = shift == "Night Shift"
    display_order = list(reversed(machine_order))   # worst=top, best=bottom
    fig           = go.Figure()
    seen_legend   = set()

    for machine in machine_order:
        rows = sdf[sdf["Machine_ID"] == machine].sort_values("Start_Time")
        for _, row in rows.iterrows():
            status = row["Status"]
            color  = STATUS_COLOR.get(status, "#888888")
            t0     = _to_chart_hour(
                row["Start_Time"].hour + row["Start_Time"].minute / 60, is_night)
            t1     = _to_chart_hour(
                row["End_Time"].hour   + row["End_Time"].minute   / 60, is_night)
            if t1 <= t0:
                t1 += 24

            first = status not in seen_legend
            if first:
                seen_legend.add(status)

            fig.add_trace(go.Bar(
                x=[t1 - t0], y=[machine], base=t0,
                orientation="h",
                marker_color=color, marker_line_width=0, width=0.55,
                name=STATUS_LABEL.get(status, status),
                showlegend=first, legendgroup=status,
                hovertemplate=(
                    f"<b>{machine}</b><br>"
                    f"Status: {STATUS_LABEL.get(status, status)}<br>"
                    f"Start: {row['Start_Time'].strftime('%H:%M')}<br>"
                    f"End: {row['End_Time'].strftime('%H:%M')}<br>"
                    f"Duration: {row['Duration_Min']} min<extra></extra>"
                ),
            ))

    _add_boundary_lines(fig, sdf, machine_order, is_night)

    if is_night:
        x_min, x_max = 19, 31
        tick_vals    = list(range(19, 32))
    else:
        x_min, x_max = 7, 19
        tick_vals    = list(range(7, 20))
    tick_text = [f"{h % 24:02d}" for h in tick_vals]

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#2a2a2a", plot_bgcolor="#2a2a2a",
        font=dict(color="#cbd5e1", size=11),
        margin=dict(l=70, r=10, t=10, b=28), height=260,
        showlegend=False,   # legend shown as HTML strip above the chart
        xaxis=dict(
            range=[x_min, x_max],
            tickvals=tick_vals, ticktext=tick_text,
            showgrid=False, zeroline=False,
            tickfont=dict(size=10, color="#6b7280"),
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=display_order,
            showgrid=False,
            tickfont=dict(size=11, color="#e2e8f0"),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def build_summary_points(metrics: dict, shift: str) -> list[tuple]:
    s     = metrics["stats"]
    best  = max(s, key=lambda m: s[m]["util_pct"])
    worst = min(s, key=lambda m: s[m]["util_pct"])
    top_f = max(s, key=lambda m: s[m]["fault"])
    top_p = max(s, key=lambda m: s[m]["pm"])
    return [
        ("🟢", f"<b>All 6 machines</b> were dominant running on {shift}."),
        ("📊", f"<b>Average utilisation: {metrics['avg_util']}%</b> (UP_PRODUCT only)."),
        ("🏆", f"<b>Best performer:</b> {best} — {s[best]['util_pct']}% uptime ({s[best]['run']} min)."),
        ("⚠️", f"<b>Worst performer:</b> {worst} — {s[worst]['util_pct']}% uptime ({s[worst]['run']} min)."),
        ("🔧", f"<b>Most fault time:</b> {top_f} — {s[top_f]['fault']} min ({s[top_f]['fault_pct']}% of shift)."),
        ("🔩", f"<b>Most PM time:</b> {top_p} — {s[top_p]['pm']} min in planned maintenance."),
        ("📉", f"<b>Total downtime</b> (fault + PM): {metrics['total_down']} min across all machines."),
        ("💤", f"<b>Total idle time:</b> {metrics['total_idle']} min across all machines."),
    ]


def build_priority_items(metrics: dict) -> list[dict]:
    s, items = metrics["stats"], []
    for m, v in sorted(s.items(), key=lambda x: x[1]["fault"], reverse=True):
        if v["fault"] > 0:
            reason = f"; {v['reasons'][0]}" if v["reasons"] else ""
            items.append(dict(
                m=m, dot="r" if v["fault_pct"] >= 20 else "b",
                desc=(f"{v['fault']} min fault ({v['fault_pct']}%)"
                      f"{reason} — inspect before next shift"),
            ))
    for m, v in sorted(s.items(), key=lambda x: x[1]["pm"], reverse=True):
        if v["pm"] > 60 and not any(i["m"] == m for i in items):
            items.append(dict(
                m=m, dot="b",
                desc=f"{v['pm']} min in PM — confirm maintenance schedule",
            ))
    ok = [m for m, v in s.items() if v["fault"] == 0 and v["pm"] == 0]
    if ok:
        items.append(dict(m=", ".join(ok), dot="g",
                          desc="Strong shift — standard handover only"))
    return items[:5]


# ─────────────────────────────────────────────────────────────────────────────
#  HTML FRAGMENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _gantt_legend_html() -> str:
    """Compact inline legend strip rendered above each Gantt chart."""
    items = "".join(
        f'<span class="gantt-legend-item">'
        f'<span class="gantt-legend-swatch" style="background:{STATUS_COLOR[k]}"></span>'
        f'{label}</span>'
        for k, label in GANTT_LEGEND_ENTRIES
    )
    return f'<div class="gantt-legend">{items}</div>'


def _kpi_table_html(avg: int, total_down: int,
                    total_fault: int, total_idle: int,
                    machines_running: int = 6) -> str:
    util_color = "#4CAF50" if avg >= 80 else "#FFC107" if avg >= 50 else "#FF5252"
    return (
        f'<table class="kpi-table"><tr>'
        f'<td class="kpi-header">Machines<br>Running'
        f'  <div class="kpi-val"><span class="c-green">{machines_running}</span> of 6</div></td>'
        f'<td class="kpi-header">Average<br>Utilization'
        f'  <div class="kpi-val" style="color:{util_color}">{avg}%</div></td>'
        f'<td class="kpi-header">Total Shift<br>Downtime'
        f'  <div class="kpi-val c-yellow">{total_down} min</div></td>'
        f'<td class="kpi-header">Total (Fault)<br>Repair Time'
        f'  <div class="kpi-val c-red">{total_fault} min</div></td>'
        f'<td class="kpi-header">Total<br>Idle Time'
        f'  <div class="kpi-val c-blue">{total_idle} min</div></td>'
        f'</tr></table>'
        f'<div class="colour-key">'
        f'80%–100% = Green &nbsp;|&nbsp; 50%–80% = Yellow &nbsp;|&nbsp; &lt;50% = Red'
        f'</div>'
    )


def _mini_bar(run_p: int, fault_p: int, pm_p: int, idle_p: int) -> str:
    other = max(0, 100 - run_p - fault_p - pm_p - idle_p)
    segs  = [
        (run_p,   STATUS_COLOR["UP_PRODUCT"]),
        (fault_p, STATUS_COLOR["IN_REPAIR"]),
        (pm_p,    STATUS_COLOR["IN_PM"]),
        (idle_p,  STATUS_COLOR["IDLE"]),
        (other,   "#374151"),
    ]
    inner = "".join(
        f'<div style="flex:{p};background:{c};height:100%"></div>'
        for p, c in segs if p > 0
    )
    return f'<div class="mcard-bar">{inner}</div>'


def _btn_border(cfg: dict, is_active: bool) -> str:
    return (f"3px solid {cfg['border_active']}" if is_active
            else f"1px solid {cfg['border_idle']}")


def _snap_tile_html(label: str, day_val: int, night_val: int,
                    pct_change) -> str:
    """
    Comparison tile for the AI Advisor snapshot row.
    Shows Day minutes, Night minutes, and % change between shifts.
    No arrows — just plain numbers and colour-coded % badge.
    pct_change = float | None (None means both values are 0).
    """
    if pct_change is None:
        pct_html = '<span style="font-size:11px;color:#6b7280">no change</span>'
    else:
        sign  = "+" if pct_change > 0 else ""
        pct_html = (
            f'<span style="font-size:12px;font-weight:700;color:#2196F3">'
            f'{sign}{pct_change}%</span>'
        )

    return (
        f'<div class="snap-tile">'
        f'  <div class="snap-label">{label}</div>'
        f'  <div style="display:flex;justify-content:space-between;'
        f'             align-items:center;margin-top:8px;gap:6px">'
        f'    <div style="text-align:center;flex:1">'
        f'      <div style="font-size:10px;color:#6b7280;margin-bottom:2px">Day</div>'
        f'      <div class="snap-value">{day_val}'
        f'        <span style="font-size:11px;color:#9ca3af">min</span>'
        f'      </div>'
        f'    </div>'
        f'    <div style="text-align:center;flex:1">'
        f'      <div style="font-size:10px;color:#6b7280;margin-bottom:2px">Night</div>'
        f'      <div class="snap-value">{night_val}'
        f'        <span style="font-size:11px;color:#9ca3af">min</span>'
        f'      </div>'
        f'    </div>'
        f'    <div style="text-align:center;flex:1">'
        f'      <div style="font-size:10px;color:#6b7280;margin-bottom:2px">Change</div>'
        f'      <div style="margin-top:4px">{pct_html}</div>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_kpi_table(avg: int, total_down: int,
                     total_fault: int, total_idle: int) -> None:
    st.markdown(
        _kpi_table_html(avg, total_down, total_fault, total_idle),
        unsafe_allow_html=True,
    )


def render_shift_buttons(active: str) -> None:
    """Three colour-coded view buttons below the KPI summary."""
    col1, col2, col3, _ = st.columns([2, 2, 2, 4])
    with col1:
        if st.button("🌤  Day 07:00–19:00\n720 min",
                     key="btn_day", use_container_width=True):
            st.session_state["view"] = "Day Shift"
            st.rerun()
    with col2:
        if st.button("🌙  Night 19:00–07:00\n720 min",
                     key="btn_night", use_container_width=True):
            st.session_state["view"] = "Night Shift"
            st.rerun()
    with col3:
        if st.button("⚖  Compare Both Shifts",
                     key="btn_compare", use_container_width=True):
            st.session_state["view"] = "Compare"
            st.rerun()

    st.markdown(f"""
<style>
div[data-testid="stColumns"]>div:nth-child(1) button{{
    background:{BTN_DAY['bg']}!important;color:#fff!important;
    border:{_btn_border(BTN_DAY,   active=='Day Shift'  )}!important;
    font-weight:800!important;border-radius:6px!important}}
div[data-testid="stColumns"]>div:nth-child(2) button{{
    background:{BTN_NIGHT['bg']}!important;color:#fff!important;
    border:{_btn_border(BTN_NIGHT, active=='Night Shift')}!important;
    font-weight:800!important;border-radius:6px!important}}
div[data-testid="stColumns"]>div:nth-child(3) button{{
    background:{BTN_COMPARE['bg']}!important;color:#fff!important;
    border:{_btn_border(BTN_COMPARE, active=='Compare'  )}!important;
    font-weight:800!important;border-radius:6px!important}}
</style>""", unsafe_allow_html=True)


def render_gantt_card(df: pd.DataFrame, shift: str, metrics: dict) -> None:
    """
    Gantt card with:
     - Status legend strip directly above the chart
     - Machines ordered worst (top) → best (bottom)
    """
    order = worst_to_best(metrics)
    st.markdown(
        '<div style="background:#2a2a2a;border-radius:10px;'
        'padding:14px 16px 6px 16px;margin-bottom:14px;border:1px solid #374151">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-label">12-HOUR TIMELINE — worst (top) → best (bottom)</div>',
        unsafe_allow_html=True,
    )
    # Legend strip directly above the chart
    st.markdown(_gantt_legend_html(), unsafe_allow_html=True)
    st.plotly_chart(
        build_timeline(df, shift, order),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_summary_card(metrics: dict, shift: str) -> None:
    rows_html = "".join(
        f'<div style="display:flex;gap:8px;margin-bottom:7px;'
        f'font-size:12.5px;color:#cbd5e1;line-height:1.5">'
        f'<span style="flex-shrink:0">{icon}</span><span>{text}</span></div>'
        for icon, text in build_summary_points(metrics, shift)
    )
    st.markdown(
        f'<div class="summary-card">'
        f'<span class="summary-badge">SHIFT SUMMARY</span>'
        f'<div style="margin-top:6px">{rows_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_priorities_card(metrics: dict) -> None:
    dot_cls    = {"r": "dot-r", "b": "dot-b", "g": "dot-g"}
    items_html = "".join(
        f'<div class="pri-item">'
        f'<div class="{dot_cls.get(p["dot"], "dot-b")}"></div>'
        f'<div><b>{p["m"]}</b> — {p["desc"]}</div>'
        f'</div>'
        for p in build_priority_items(metrics)
    )
    st.markdown(
        f'<div class="summary-card">'
        f'<div class="section-label">PRIORITIES FOR NEXT SHIFT</div>'
        f'{items_html}</div>',
        unsafe_allow_html=True,
    )


def render_machine_cards(metrics: dict) -> None:
    """Machine cards in worst-first order (top-left = lowest util%)."""
    order = worst_to_best(metrics)
    cols  = st.columns(3, gap="small")
    for i, machine in enumerate(order):
        v = metrics["stats"][machine]
        concern_html = (
            f'<span class="concern-badge {v["concern_cls"]}">{v["concern"]}</span>'
            if v["concern"] else ""
        )
        with cols[i % 3]:
            st.markdown(
                f'<div class="mcard">'
                f'<div class="mcard-header">'
                f'  <div>'
                f'    <div class="mcard-id">{machine}</div>'
                f'    <div class="mcard-type">{MACHINE_TYPE.get(machine, "")}</div>'
                f'  </div>'
                f'  <span class="mcard-badge">running</span>'
                f'</div>'
                f'{_mini_bar(v["util_pct"], v["fault_pct"], v["pm_pct"], v["idle_pct"])}'
                f'<div class="mcard-stats">'
                f'  Up <b>{v["util_pct"]}%</b> &nbsp;·&nbsp; '
                f'  Downtime <b>{v["downtime"]}m</b> &nbsp;·&nbsp; '
                f'  Fault <b>{v["fault"]}m</b>'
                f'</div>'
                f'{concern_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  AI ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
def _ai_prompt(machine: str, shift: str, v: dict) -> str:
    reasons = ", ".join(v["reasons"]) if v["reasons"] else "none recorded"
    return (
        f"You are a semiconductor fab equipment engineer.\n"
        f"Machine {machine} just completed the {shift}.\n\n"
        f"Performance data:\n"
        f"  - Running (UP_PRODUCT): {v['run']} min ({v['util_pct']}%)\n"
        f"  - Fault/repair time:    {v['fault']} min ({v['fault_pct']}%)\n"
        f"  - PM time:              {v['pm']} min ({v['pm_pct']}%)\n"
        f"  - Idle time:            {v['idle']} min ({v['idle_pct']}%)\n"
        f"  - Total downtime:       {v['downtime']} min\n"
        f"  - Downtime reasons:     {reasons}\n\n"
        f"Give exactly 3 clear, specific actions in plain English the team should "
        f"take before the next shift. Number them 1, 2, 3. "
        f"Max 2 sentences each. No jargon. No preamble."
    )


def _call_ai(prompt: str, provider: str, api_key: str, model: str) -> str:
    """Dispatch to the selected AI provider and return response text."""
    if provider == "OpenAI (ChatGPT)":
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model or "gpt-4o-mini",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 400},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    if provider == "Anthropic (Claude)":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key,
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json={"model": model or "claude-haiku-4-5-20251001",
                  "max_tokens": 400,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()

    if provider == "Google Gemini":
        mid = model or "gemini-1.5-flash"
        r   = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{mid}:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    if provider == "Ollama (Local / Free)":
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model or "llama3", "prompt": prompt, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    return "Provider not recognised."


def _rule_based_actions(machine: str, v: dict) -> list[str]:
    """Three rule-based actions when no AI provider is configured."""
    actions = []

    if v["fault"] >= 120:
        actions.append(
            f"**Escalate fault investigation on {machine}.** "
            f"It had {v['fault']} min of fault/repair ({v['fault_pct']}% of shift) — "
            f"assign a senior technician to find the root cause before next shift."
        )
    elif v["fault"] > 0:
        actions.append(
            f"**Review repair logs for {machine}.** "
            f"It logged {v['fault']} min of fault — verify the fix is complete "
            f"and confirm the fault code does not recur at shift start."
        )
    else:
        actions.append(
            f"**Run a health check on {machine} at handover.** "
            f"No fault was recorded — a quick functional check ensures continued good health."
        )

    if v["pm"] >= 90:
        actions.append(
            f"**Get PM sign-off for {machine}.** "
            f"It spent {v['pm']} min in planned maintenance — "
            f"confirm all checklist items are signed before returning to production."
        )
    elif v["pm"] > 0:
        actions.append(
            f"**Confirm PM schedule for {machine}.** "
            f"The machine had {v['pm']} min of PM — "
            f"check the next window is aligned to avoid production overlap."
        )
    else:
        actions.append(
            f"**Check upcoming PM due date for {machine}.** "
            f"No PM was performed this shift — confirm the next date and pre-arrange parts."
        )

    if v["idle"] >= 60:
        actions.append(
            f"**Investigate {machine} idle time.** "
            f"The machine was idle {v['idle']} min ({v['idle_pct']}%) — "
            f"determine whether the cause was missing work orders, staffing, or a bottleneck."
        )
    else:
        actions.append(
            f"**Pre-load the next job for {machine}.** "
            f"Utilisation was {v['util_pct']}% — prepare the next work order and materials "
            f"before shift start to minimise ramp-up time."
        )

    return actions[:3]


def render_ai_advisor(ai_provider: str, ai_key: str, ai_model: str,
                      day_m: dict, night_m: dict) -> None:
    """
    AI Advisor section.
    Snapshot tiles show Day vs Night values with:
      ▲ green  = night value increased vs day
      ▼ red    = night value decreased vs day
      (none)   = no change
    """
    st.markdown("---")
    st.markdown("## 🤖 Machine AI Advisor")
    st.markdown(
        "Select a machine to see Day vs Night shift comparison and receive "
        "**3 specific plain-English actions** recommended before the next shift."
    )

    col_m, col_s = st.columns([2, 1])
    with col_m:
        machine = st.selectbox("Select Machine", MACHINES, key="adv_machine")
    with col_s:
        adv_shift = st.selectbox("Get actions for shift",
                                 ["Day Shift", "Night Shift"], key="adv_shift")

    # Build shift-comparison arrows
    comparison = compare_shifts(day_m, night_m, machine)

    # ── Snapshot comparison tiles ─────────────────────────────────────────────
    tile_specs = [
        ("Running",  "run"),
        ("Fault",    "fault"),
        ("PM",       "pm"),
        ("Idle",     "idle"),
        ("Downtime", "downtime"),
    ]
    cols = st.columns(len(tile_specs), gap="small")
    for col, (label, field) in zip(cols, tile_specs):
        c = comparison[field]
        with col:
            st.markdown(
                _snap_tile_html(label, c["day"], c["night"], c["pct_change"]),
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="font-size:10px;color:#6b7280;margin:6px 0 14px 0">'
        '<span style="color:#2196F3">%</span> = change in minutes between Day and Night shifts &nbsp;|&nbsp; '
        'no change = identical minutes'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── AI action recommendations ─────────────────────────────────────────────
    metrics = day_m if adv_shift == "Day Shift" else night_m
    v       = metrics["stats"][machine]

    if ai_provider == "None (no AI)":
        actions = _rule_based_actions(machine, v)
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ai-box-title">📋 Recommended Actions — '
            f'{machine} / {adv_shift} (rule-based)</div>',
            unsafe_allow_html=True,
        )
        for i, a in enumerate(actions, 1):
            st.markdown(
                f'<div class="ai-action"><b>Action {i}:</b> {a}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("💡 Connect an AI provider in the sidebar for AI-generated recommendations.")

    else:
        if st.button(f"🤖 Get AI Recommendations for {machine}", type="primary"):
            if not ai_key and ai_provider != "Ollama (Local / Free)":
                st.warning("Enter your API key in the sidebar first.")
            else:
                with st.spinner(f"Asking {ai_provider}…"):
                    try:
                        text    = _call_ai(
                            _ai_prompt(machine, adv_shift, v),
                            ai_provider, ai_key, ai_model,
                        )
                        actions = [
                            p.strip()
                            for p in re.split(r'\n?\s*\d+[\.\)]\s+', text.strip())
                            if p.strip()
                        ]
                        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="ai-box-title">📋 AI Recommendations — '
                            f'{machine} / {adv_shift} ({ai_provider})</div>',
                            unsafe_allow_html=True,
                        )
                        for i, a in enumerate(actions[:3], 1):
                            st.markdown(
                                f'<div class="ai-action"><b>Action {i}:</b> {a}</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                    except requests.exceptions.ConnectionError:
                        st.error("Connection failed. Check network or Ollama is running.")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"API error {e.response.status_code}: "
                                 f"check key / model name.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload shift_data.xlsx", type=["xlsx"])
    st.markdown("---")
    st.markdown("### 🤖 AI Advisor Settings")
    ai_provider = st.selectbox("AI Provider", [
        "None (no AI)",
        "OpenAI (ChatGPT)",
        "Anthropic (Claude)",
        "Google Gemini",
        "Ollama (Local / Free)",
    ])
    ai_key = ai_model = ""
    if ai_provider not in ("None (no AI)", "Ollama (Local / Free)"):
        ai_key   = st.text_input("API Key", type="password",
                                 help="Used this session only — never stored.")
        ai_model = st.text_input("Model (optional)", placeholder={
            "OpenAI (ChatGPT)":   "gpt-4o-mini",
            "Anthropic (Claude)": "claude-haiku-4-5-20251001",
            "Google Gemini":      "gemini-1.5-flash",
        }.get(ai_provider, ""))
    if ai_provider == "Ollama (Local / Free)":
        ai_model = st.text_input("Ollama model", placeholder="llama3")
        st.caption("Requires Ollama running locally on port 11434.")


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
data_source = uploaded if uploaded else "shift_dummy_data2.xlsx"
try:
    df = load_data(data_source)
except Exception as e:
    st.error(
        "Could not load data. "
        "Place 'shift_dummy_data2.xlsx' in the same folder.\n\n"
        f"{e}"
    )
    st.stop()

day_m, night_m = calc_metrics(df, "Day Shift"), calc_metrics(df, "Night Shift")
today_str      = date.today().strftime("%d %B %Y")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Title + byline ─────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="fab-title">FabSight – Shift Performance Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="fab-byline">By Uptime Guardians</div>',
            unsafe_allow_html=True)

# ── 2. Date + combined KPI table ─────────────────────────────────────────────
st.markdown(f'<div class="date-line">Date: <u>{today_str}</u></div>',
            unsafe_allow_html=True)
render_kpi_table(
    avg         = (day_m["avg_util"]    + night_m["avg_util"])    // 2,
    total_down  =  day_m["total_down"]  + night_m["total_down"],
    total_fault =  day_m["total_fault"] + night_m["total_fault"],
    total_idle  =  day_m["total_idle"]  + night_m["total_idle"],
)

# ── 3. Separator ──────────────────────────────────────────────────────────────
st.markdown("---")

# ── 4. Shift selector buttons (below the summary dashboard) ───────────────────
if "view" not in st.session_state:
    st.session_state["view"] = "Compare"

render_shift_buttons(st.session_state["view"])
view = st.session_state["view"]
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN VIEWS
# ─────────────────────────────────────────────────────────────────────────────
if view == "Compare":
    # ── Side-by-side shift comparison ─────────────────────────────────────────
    col_d, col_n = st.columns(2, gap="medium")
    for col, shift, pill_cls, metrics in [
        (col_d, "Day Shift",   "pill-day",   day_m),
        (col_n, "Night Shift", "pill-night", night_m),
    ]:
        label = ("Day shift · 07:00–19:00"
                 if shift == "Day Shift" else "Night shift · 19:00–07:00")
        with col:
            st.markdown(f'<div class="shift-pill {pill_cls}">{label}</div>',
                        unsafe_allow_html=True)
            render_gantt_card(df, shift, metrics)
            render_summary_card(metrics, shift)
            render_priorities_card(metrics)

else:
    # ── Single-shift detail view ───────────────────────────────────────────────
    metrics  = day_m   if view == "Day Shift" else night_m
    pill_cls = "pill-day" if view == "Day Shift" else "pill-night"
    label    = ("Day shift · 07:00–19:00"
                if view == "Day Shift" else "Night shift · 19:00–07:00")

    st.markdown(f'<div class="shift-pill {pill_cls}">{label}</div>',
                unsafe_allow_html=True)
    render_kpi_table(
        metrics["avg_util"], metrics["total_down"],
        metrics["total_fault"], metrics["total_idle"],
    )
    render_gantt_card(df, view, metrics)

    st.markdown('<div class="section-label">MACHINE HEALTH — WORST TO BEST PERFORMANCE</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Cards ordered worst → best (left to right) &nbsp;·&nbsp; '
        'bar = running / fault / PM / idle &nbsp;·&nbsp; badge = top concern'
        '</div>',
        unsafe_allow_html=True,
    )
    render_machine_cards(metrics)

    col_s, col_p = st.columns(2, gap="medium")
    with col_s:
        render_summary_card(metrics, view)
    with col_p:
        render_priorities_card(metrics)


# ─────────────────────────────────────────────────────────────────────────────
#  AI ADVISOR  (always at the bottom)
# ─────────────────────────────────────────────────────────────────────────────
render_ai_advisor(ai_provider, ai_key, ai_model, day_m, night_m)
