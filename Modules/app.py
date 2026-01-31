import os
import json
import time
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(".")
MAIN_FILE = PROJECT_ROOT / "maintwo.py"

CONTROL_FILE = PROJECT_ROOT / "control.json"
STATUS_FILE = PROJECT_ROOT / "status.json"
SUMMARY_FILE = PROJECT_ROOT / "summary.json"


def format_seconds(seconds: int) -> str:
    seconds = max(0, int(seconds or 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"


def write_control(command: str):
    payload = {"command": command, "ts": time.time()}
    with open(CONTROL_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def donut_chart(attentive, distracted, away):
    labels = ["Attentive", "Distracted", "Away"]
    values = [attentive, distracted, away]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        showlegend=False,
    )
    return fig


def is_running():
    proc = st.session_state.get("proc")
    return proc is not None and proc.poll() is None


def start_backend():
    if not MAIN_FILE.exists():
        st.error("main.py not found. Put app.py in the same folder.")
        return

    # Start main.py as subprocess
    proc = subprocess.Popen(
        ["python", str(MAIN_FILE)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state.proc = proc


def kill_backend():
    proc = st.session_state.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
    st.session_state.proc = None


# ----------------------------
# UI Setup
# ----------------------------
st.set_page_config(page_title="FocusOS Dashboard", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .subtext { color: rgba(255,255,255,0.65); margin-top: -0.6rem; }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.2rem;
        border-radius: 20px;
    }
    .big-text {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.2rem 0 0 0;
    }
    .muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "proc" not in st.session_state:
    st.session_state.proc = None


# ----------------------------
# Header
# ----------------------------
st.title("FocusOS Session Control + Summary")
st.markdown('<div class="subtext">Run main.py from Streamlit and control it live.</div>', unsafe_allow_html=True)
st.divider()


# ----------------------------
# Control Row
# ----------------------------
status = safe_read_json(STATUS_FILE)
backend_running = is_running()

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

with c1:
    if st.button("🚀 Start Backend", use_container_width=True):
        if backend_running:
            st.toast("Backend already running 💀", icon="⚡")
        else:
            start_backend()
            st.toast("Backend started ✅", icon="🧠")

with c2:
    if st.button("▶️ Start Session", use_container_width=True):
        if not backend_running:
            st.error("Start backend first.")
        else:
            write_control("START_SESSION")
            st.toast("Sent START_SESSION ✅", icon="✅")

with c3:
    if st.button("🎯 Calibrate", use_container_width=True):
        if not backend_running:
            st.error("Start backend first.")
        else:
            write_control("CALIBRATE")
            st.toast("Sent CALIBRATE 🎯", icon="🎯")

with c4:
    if st.button("⛔ End Session", use_container_width=True):
        if not backend_running:
            st.error("Start backend first.")
        else:
            write_control("END_SESSION")
            st.toast("Sent END_SESSION 🏁", icon="🏁")

with c5:
    if st.button("🧨 Kill Backend", use_container_width=True):
        kill_backend()
        st.toast("Backend killed 💀", icon="💀")


# Status box
st.markdown("### Backend Status")
if backend_running:
    st.success("🟢 Running")
else:
    st.error("🔴 Not running")

if status:
    st.caption(f"State: `{status.get('state','?')}` | {status.get('message','')}")
else:
    st.caption("No status.json yet (backend hasn’t written it).")

st.divider()


# ----------------------------
# Summary Section
# ----------------------------
st.header("FocusOS Session Summary")
st.markdown('<div class="subtext">Post-session report</div>', unsafe_allow_html=True)

summary = safe_read_json(SUMMARY_FILE)

if summary is None:
    st.info("No summary.json yet. Run a session and end it.")

    # Auto refresh while backend running
    if backend_running:
        time.sleep(1.2)
        st.rerun()

    st.stop()

# Read fields
session_start = summary.get("session_start", "N/A")
session_end = summary.get("session_end", "N/A")
total_seconds = int(summary.get("total_seconds", 0))

att = int(summary.get("attentive_seconds", 0))
dis = int(summary.get("distracted_seconds", 0))
away = int(summary.get("away_seconds", 0))

focus_percent = float(summary.get("focus_percent", 0.0))
focus_percent = max(0.0, min(100.0, focus_percent))

csv_path = summary.get("csv_path", "dashboard.csv")
csv_path = Path(csv_path)

# Cards
st.subheader("Session Timing")
t1, t2, t3 = st.columns(3)

with t1:
    st.markdown(f"""
    <div class="card">
      <h4>Start Time</h4>
      <div class="big-text">🕒</div>
      <div class="muted">{session_start}</div>
    </div>
    """, unsafe_allow_html=True)

with t2:
    st.markdown(f"""
    <div class="card">
      <h4>End Time</h4>
      <div class="big-text">🏁</div>
      <div class="muted">{session_end}</div>
    </div>
    """, unsafe_allow_html=True)

with t3:
    st.markdown(f"""
    <div class="card">
      <h4>Total Time</h4>
      <div class="big-text">{format_seconds(total_seconds)}</div>
      <div class="muted">Total tracked duration</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

st.subheader("Breakdown")
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown(f"""
    <div class="card">
      <h4>✅ Attentive</h4>
      <div class="big-text">{format_seconds(att)}</div>
      <div class="muted">Locked in mode</div>
    </div>
    """, unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="card">
      <h4>❌ Distracted</h4>
      <div class="big-text">{format_seconds(dis)}</div>
      <div class="muted">Brain alt-tabbed 💀</div>
    </div>
    """, unsafe_allow_html=True)

with b3:
    st.markdown(f"""
    <div class="card">
      <h4>🚫 Away</h4>
      <div class="big-text">{format_seconds(away)}</div>
      <div class="muted">Not in frame</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Focus Score")
    st.metric("Focus %", f"{focus_percent:.1f}%")
    st.progress(focus_percent / 100.0)
    st.caption(f"Dashboard saved to: `{csv_path}`")

with right:
    st.subheader("Distribution")
    st.plotly_chart(donut_chart(att, dis, away), use_container_width=True)

st.divider()

st.subheader("Export")

# Download CSV
if csv_path.exists():
    with open(csv_path, "rb") as f:
        st.download_button(
            "⬇️ Download CSV",
            data=f,
            file_name=csv_path.name,
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.warning("CSV not found. main.py didn’t save it.")

# Preview
df = safe_read_csv(csv_path)
if df is not None:
    st.markdown("**CSV Preview**")
    st.dataframe(df, use_container_width=True, height=260)
