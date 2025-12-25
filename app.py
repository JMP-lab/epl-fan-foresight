# app.py
# ============================================================
# EPL Fan Forecast — Streamlit MVP (single-screen, human-friendly)
#
# ✅ Big “prediction card” (sporty gradient) with the question FIRST
# ✅ Embedded YouTube video + real clickable link
# ✅ Votes persisted (votes.csv) + live Crowd leaning (YES share)
# ✅ Temperature ruler with dot marker
# ✅ Momentum preview (Early vs Now) + Δ momentum ruler with dot marker
# ✅ Rename scary terms:
#    - “Settlement” -> “Final Whistle”
#    - “Market price” -> “Crowd leaning”
#    - “Reputation leaderboard” -> “Fan prediction score”
# ✅ Channel/title resolution:
#    - If video_meta.csv exists: uses it
#    - Else: tries YouTube API (YOUTUBE_API_KEY)
#    - Else: falls back gracefully (no crash)
#
# Required:
#   data_processed/video_features_with_trajectory.csv (preferred) OR
#   data_processed/video_features.csv (fallback)
# Required for clock:
#   data_processed/{video_id}_comments.csv  (must contain published_at)
# Optional:
#   data_processed/video_meta.csv
# ============================================================

import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------
# PATHS
# ----------------------------
# Prefer "folder of this app.py" so you can run it anywhere
APP_DIR = Path(__file__).resolve().parent

# If you want to hard-pin, uncomment the next line
# APP_DIR = Path(r"C:\Users\JMPark\Documents\epl_fan_forecast")

DATA_DIR = APP_DIR / "data_processed"

FEATURES_PATH = DATA_DIR / "video_features_with_trajectory.csv"
FALLBACK_FEATURES_PATH = DATA_DIR / "video_features.csv"
META_PATH = DATA_DIR / "video_meta.csv"       # optional
VOTES_PATH = DATA_DIR / "votes.csv"           # created automatically
SCORES_PATH = DATA_DIR / "scores.csv"         # created automatically

# ----------------------------
# LABELS / HELPERS
# ----------------------------
def ecosystem_state(temp: float) -> str:
    if pd.isna(temp):
        return "Unknown"
    if temp < 2.20:
        return "🧊 Calm"
    if temp < 2.32:
        return "🌤 Active"
    if temp < 2.42:
        return "🔥 Tense"
    return "🌪 Chaotic"

def momentum_label(delta: float) -> str:
    if pd.isna(delta):
        return "Unknown"
    if delta <= -0.20:
        return "📉 Cooling"
    if delta >= 0.20:
        return "📈 Heating up"
    return "➖ Steady"

def state_explainer(state: str) -> str:
    return {
        "🧊 Calm": "Low activation and limited back-and-forth.",
        "🌤 Active": "Healthy discussion with moderate interaction.",
        "🔥 Tense": "Heavier debate and stronger intensity signals.",
        "🌪 Chaotic": "High participation + deep replies + tension signals.",
    }.get(state, "Not enough data to classify.")

def momentum_explainer(mom: str) -> str:
    return {
        "📉 Cooling": "The discussion cools after the early window.",
        "➖ Steady": "No meaningful change after the early window.",
        "📈 Heating up": "The discussion intensifies after the early window.",
    }.get(mom, "Not enough data to classify.")

def fmt_timedelta(td: pd.Timedelta) -> str:
    if td is None or pd.isna(td):
        return "NA"
    secs = int(td.total_seconds())
    if secs < 0:
        secs = 0
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"

def gradient_by_temp(temp: float) -> str:
    if pd.isna(temp):
        return "linear-gradient(135deg, #263238 0%, #455a64 100%)"
    if temp < 2.20:  # Calm
        return "linear-gradient(135deg, #0b3d91 0%, #1976d2 55%, #64b5f6 100%)"
    if temp < 2.32:  # Active
        return "linear-gradient(135deg, #0f6d3b 0%, #2e7d32 55%, #a5d6a7 100%)"
    if temp < 2.42:  # Tense
        return "linear-gradient(135deg, #e65100 0%, #fb8c00 60%, #ffe0b2 100%)"
    return "linear-gradient(135deg, #3a0d68 0%, #ad1457 55%, #d32f2f 100%)"

def bar_line(label: str, value: float, lo=2.0, hi=2.6, width=18) -> str:
    if value is None or math.isnan(value):
        return f"{label:<14}  {'NA':>6}"
    x = (value - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    filled = int(round(x * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{label:<14}  {bar}  {value:.3f}"

# ----------------------------
# DATA LOADING
# ----------------------------
@st.cache_data
def load_features():
    if FEATURES_PATH.exists():
        df = pd.read_csv(FEATURES_PATH)
    elif FALLBACK_FEATURES_PATH.exists():
        df = pd.read_csv(FALLBACK_FEATURES_PATH)
    else:
        return pd.DataFrame()

    # Ensure columns exist so UI never breaks
    if "video_id" not in df.columns:
        return pd.DataFrame()

    if "temperature_v1" not in df.columns:
        df["temperature_v1"] = pd.NA

    if "temp_early_48h" not in df.columns:
        df["temp_early_48h"] = pd.NA

    if "temp_full_recomputed" not in df.columns:
        df["temp_full_recomputed"] = df.get("temperature_v1", pd.NA)

    if "delta_temp_48h" not in df.columns:
        df["delta_temp_48h"] = df["temp_full_recomputed"] - df["temp_early_48h"]

    return df

@st.cache_data
def load_meta_csv():
    if not META_PATH.exists():
        return pd.DataFrame(columns=["video_id", "title", "channel_title", "video_published_at"])

    m = pd.read_csv(META_PATH)

    # Normalize common column names
    rename_map = {}
    if "channelTitle" in m.columns and "channel_title" not in m.columns:
        rename_map["channelTitle"] = "channel_title"
    if "publishedAt" in m.columns and "video_published_at" not in m.columns:
        rename_map["publishedAt"] = "video_published_at"
    if rename_map:
        m = m.rename(columns=rename_map)

    if "video_id" not in m.columns:
        return pd.DataFrame(columns=["video_id", "title", "channel_title", "video_published_at"])

    # Ensure consistent columns
    for c in ["title", "channel_title", "video_published_at"]:
        if c not in m.columns:
            m[c] = pd.NA

    return m[["video_id", "title", "channel_title", "video_published_at"]].copy()

@st.cache_data
def fetch_meta_from_api(video_id: str):
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        return {"title": None, "channel_title": None, "video_published_at": None}

    try:
        from googleapiclient.discovery import build  # import inside (safe)
        yt = build("youtube", "v3", developerKey=api_key)
        res = yt.videos().list(part="snippet", id=video_id).execute()
        items = res.get("items", [])
        if not items:
            return {"title": None, "channel_title": None, "video_published_at": None}
        sn = items[0].get("snippet", {})
        return {
            "title": sn.get("title"),
            "channel_title": sn.get("channelTitle"),
            "video_published_at": sn.get("publishedAt"),
        }
    except Exception:
        return {"title": None, "channel_title": None, "video_published_at": None}

def get_market_clock(video_id: str):
    """
    Final Whistle at t0 + 48h, where t0 is first comment timestamp.
    Requires: data_processed/{video_id}_comments.csv with 'published_at'
    """
    c_path = DATA_DIR / f"{video_id}_comments.csv"
    if not c_path.exists():
        return None, None, None, str(c_path)

    dfc = pd.read_csv(c_path)
    if "published_at" not in dfc.columns or dfc.empty:
        return None, None, None, str(c_path)

    t0 = pd.to_datetime(dfc["published_at"], errors="coerce").min()
    if pd.isna(t0):
        return None, None, None, str(c_path)

    whistle_at = t0 + pd.Timedelta(hours=48)
    remaining = whistle_at - pd.Timestamp.utcnow()
    return t0, whistle_at, remaining, str(c_path)

# ----------------------------
# VOTES (persisted)
# ----------------------------
def ensure_user_id():
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())[:8]
    return st.session_state["user_id"]

def votes_mtime() -> float:
    return VOTES_PATH.stat().st_mtime if VOTES_PATH.exists() else 0.0

@st.cache_data
def load_votes(_mtime: float):
    if not VOTES_PATH.exists():
        return pd.DataFrame(columns=["ts_utc", "video_id", "user_id", "vote"])
    dfv = pd.read_csv(VOTES_PATH)
    for c in ["ts_utc", "video_id", "user_id", "vote"]:
        if c not in dfv.columns:
            dfv[c] = pd.NA
    return dfv[["ts_utc", "video_id", "user_id", "vote"]].copy()

def upsert_vote(video_id: str, user_id: str, vote: str):
    vote = vote.upper().strip()
    if vote not in {"YES", "NO"}:
        raise ValueError("vote must be YES or NO")

    now = datetime.now(timezone.utc).isoformat()

    if VOTES_PATH.exists():
        dfv = pd.read_csv(VOTES_PATH)
    else:
        dfv = pd.DataFrame(columns=["ts_utc", "video_id", "user_id", "vote"])

    for c in ["ts_utc", "video_id", "user_id", "vote"]:
        if c not in dfv.columns:
            dfv[c] = pd.NA

    mask = (dfv["video_id"].astype(str) == str(video_id)) & (dfv["user_id"].astype(str) == str(user_id))
    if mask.any():
        dfv.loc[mask, ["ts_utc", "vote"]] = [now, vote]
    else:
        dfv = pd.concat(
            [dfv, pd.DataFrame([{"ts_utc": now, "video_id": str(video_id), "user_id": str(user_id), "vote": vote}])],
            ignore_index=True,
        )

    dfv.to_csv(VOTES_PATH, index=False)
    load_votes.clear()

def my_current_vote(df_votes: pd.DataFrame, video_id: str, user_id: str):
    d = df_votes[
        (df_votes["video_id"].astype(str) == str(video_id))
        & (df_votes["user_id"].astype(str) == str(user_id))
    ].copy()
    if d.empty:
        return None
    d["ts_utc"] = pd.to_datetime(d["ts_utc"], errors="coerce")
    d = d.sort_values("ts_utc")
    return str(d.iloc[-1]["vote"])

def crowd_leaning(df_votes: pd.DataFrame, video_id: str):
    d = df_votes[df_votes["video_id"].astype(str) == str(video_id)].copy()
    if d.empty:
        return {"p_yes": 0.5, "p_no": 0.5, "n": 0, "n_yes": 0, "n_no": 0}
    n_yes = int((d["vote"] == "YES").sum())
    n_no = int((d["vote"] == "NO").sum())
    n = n_yes + n_no
    p_yes = (n_yes / n) if n > 0 else 0.5
    return {"p_yes": p_yes, "p_no": 1 - p_yes, "n": n, "n_yes": n_yes, "n_no": n_no}

# ----------------------------
# FINAL WHISTLE (scoring)
# ----------------------------
def scores_mtime() -> float:
    return SCORES_PATH.stat().st_mtime if SCORES_PATH.exists() else 0.0

@st.cache_data
def load_scores(_mtime: float):
    if not SCORES_PATH.exists():
        return pd.DataFrame(columns=["user_id", "score"])
    d = pd.read_csv(SCORES_PATH)
    if "user_id" not in d.columns:
        d["user_id"] = pd.NA
    if "score" not in d.columns:
        d["score"] = 0
    d["score"] = pd.to_numeric(d["score"], errors="coerce").fillna(0)
    return d[["user_id", "score"]].copy()

def save_scores(df_scores: pd.DataFrame):
    df_scores.to_csv(SCORES_PATH, index=False)
    load_scores.clear()

def compute_winner(delta_temp_48h: float) -> str:
    if pd.isna(delta_temp_48h):
        return "UNSETTLED"
    return "YES" if float(delta_temp_48h) >= 0.20 else "NO"
def simulate_votes(p_yes: float, n: int):
    p_yes = max(0.0, min(1.0, float(p_yes)))
    n = int(max(1, n))
    n_yes = int(round(p_yes * n))
    n_no = n - n_yes
    return {"p_yes": n_yes / n, "p_no": n_no / n, "n": n, "n_yes": n_yes, "n_no": n_no}
def apply_final_whistle_scoring(video_id: str, winner: str):
    if winner not in {"YES", "NO"}:
        return
    if not VOTES_PATH.exists():
        return

    dfv = pd.read_csv(VOTES_PATH)
    dfv = dfv[dfv["video_id"].astype(str) == str(video_id)].copy()
    if dfv.empty:
        return

    # One vote per user (latest)
    dfv["ts_utc"] = pd.to_datetime(dfv["ts_utc"], errors="coerce")
    dfv = dfv.sort_values("ts_utc").drop_duplicates(subset=["video_id", "user_id"], keep="last")

    df_scores = load_scores(scores_mtime())
    score_map = {str(r.user_id): float(r.score) for _, r in df_scores.iterrows()}

    for _, r in dfv.iterrows():
        uid = str(r["user_id"])
        v = str(r["vote"])
        score_map[uid] = score_map.get(uid, 0.0) + (1.0 if v == winner else -1.0)

    out = (
        pd.DataFrame([{"user_id": k, "score": v} for k, v in score_map.items()])
        .sort_values("score", ascending=False)
    )
    save_scores(out)

# ----------------------------
# STREAMLIT PAGE
# ----------------------------
st.set_page_config(page_title="EPL Fan Forecast", layout="wide")
st.title("Will the Conversation Heat Up — or Cool Down")
st.caption("From Kickoff to Chaos: Tracking post-launch momentum in Premier League discussions")

df = load_features()
meta = load_meta_csv()

if df.empty:
    st.error(
        "No features file found.\n"
        f"Expected:\n- {FEATURES_PATH}\n(or fallback: {FALLBACK_FEATURES_PATH})"
    )
    st.stop()

# Sidebar event picker
st.sidebar.header("Event")
video_ids = df["video_id"].astype(str).tolist()
selected_video = st.sidebar.selectbox("Choose a video", video_ids, index=0)

user_id = ensure_user_id()
st.sidebar.caption(f"User: {user_id}")

st.sidebar.markdown("### 🔮 What is this foresight?")
st.sidebar.markdown(
"""
You’re not predicting likes or sentiment — you’re predicting **momentum**.

We track replies, depth, reactions, and disagreement over time.

If the conversation **accelerates after the first 48 hours**, **YES** wins. 

If it stays steady or cools down, **NO** wins.

"""
)

# Votes + leaning
df_votes = load_votes(votes_mtime())
lean = crowd_leaning(df_votes, str(selected_video))
my_vote = my_current_vote(df_votes, str(selected_video), user_id)

# Pulse animation trigger (vote count changed)
prev_n = st.session_state.get("prev_votes_n")
st.session_state["prev_votes_n"] = lean["n"]
pulse = (prev_n is not None) and (lean["n"] != prev_n)
pulse_class = "pulse" if pulse else ""

# Current row
row = df.loc[df["video_id"].astype(str) == str(selected_video)].iloc[0].to_dict()

temp = float(row.get("temperature_v1")) if pd.notna(row.get("temperature_v1")) else float("nan")
temp_early = float(row.get("temp_early_48h")) if pd.notna(row.get("temp_early_48h")) else float("nan")
temp_full = float(row.get("temp_full_recomputed")) if pd.notna(row.get("temp_full_recomputed")) else temp
delta = float(row.get("delta_temp_48h")) if pd.notna(row.get("delta_temp_48h")) else float("nan")

state = ecosystem_state(temp)
mom = momentum_label(delta)
card_gradient = gradient_by_temp(temp)

# Title / channel resolution
title = str(selected_video)
channel = None
published = None

mrow = meta.loc[meta["video_id"].astype(str) == str(selected_video)]
if len(mrow):
    m = mrow.iloc[0].to_dict()
    title = m.get("title") or title
    channel = m.get("channel_title") or channel
    published = m.get("video_published_at") or published

# Fallback to API if missing
if (channel is None) or (title == str(selected_video)):
    api_m = fetch_meta_from_api(str(selected_video))
    title = api_m.get("title") or title
    channel = api_m.get("channel_title") or channel
    published = api_m.get("video_published_at") or published

if not channel:
    channel = "Unknown channel (no video_meta.csv / no API key)"

# Clock + winner (remains admin-only)
t0, whistle_at, remaining, comments_path = get_market_clock(str(selected_video))
winner = compute_winner(delta)

# ----------------------------
# MAIN LAYOUT
# ----------------------------
left, right = st.columns([2.2, 1])

with left:
    # Header
    st.subheader(title)
    cap = f"Channel: {channel}"
    if published:
        cap += f" Published: {published}"
    st.caption(cap)

    # ========= QUESTION CARD (FIRST) =========
    st.markdown(
        f"""
        <style>
        @keyframes pulseGlow {{
          0%   {{ box-shadow: 0 6px 20px rgba(0,0,0,0.20); transform: scale(1.0); }}
          50%  {{ box-shadow: 0 12px 34px rgba(255,255,255,0.28); transform: scale(1.01); }}
          100% {{ box-shadow: 0 6px 20px rgba(0,0,0,0.20); transform: scale(1.0); }}
        }}
        .pulse {{ animation: pulseGlow 0.7s ease-in-out; }}
        </style>

        <div class="{pulse_class}" style="
            padding: 20px 24px;
            border-radius: 14px;
            background: {card_gradient};
            color: white;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            margin: 12px 0 14px 0;
            border: 1px solid rgba(255,255,255,0.15);
        ">
          <div style="font-size: 26px; font-weight: 900; line-height:1.2;">
            🏟️ Will this discussion <span style="text-decoration: underline;">ESCALATE</span> after the first 48 hours?
          </div>
          <div style="margin-top:10px; font-size:15px; opacity:0.95;">
            Escalation = more replies, deeper threads, and stronger disagreement over time.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Embedded video + link
    st.video(youtube_watch_url(str(selected_video)))
    st.markdown(f"🔗 Open on YouTube: {youtube_watch_url(str(selected_video))}")

    # Current pick
    if my_vote:
        st.info(f"Your current pick: **{my_vote}** (you can change it)")

    # Vote buttons
    b1, b2 = st.columns(2)
    with b1:
        vote_yes = st.button("🔥 YES — Heats up", use_container_width=True)
    with b2:
        vote_no = st.button("🧊 NO — Stays steady / cools", use_container_width=True)

    if vote_yes:
        upsert_vote(str(selected_video), user_id, "YES")
        st.success("Saved: YES")
        st.rerun()

    if vote_no:
        upsert_vote(str(selected_video), user_id, "NO")
        st.success("Saved: NO")
        st.rerun()

    # State + Momentum text (compact)
    st.caption(f"**State:** {state} — {state_explainer(state)}")
    st.caption(f"**Momentum:** {mom} — {momentum_explainer(mom)}")

    # ========= TEMPERATURE RULER =========
    st.markdown("### 🧰 Foresight Signals: Live match stats that help you form your foresight")
    st.caption("These indicators help you decide YES or NO. Live match indicators showing where the discussion stands.")
    st.markdown("### 1. 🌡️ Conversation Temperature")
    st.caption("🧠 This answers the question: **where is the discussion right now?**")
    lo, hi = 2.0, 2.6
    safe_temp = temp if not math.isnan(temp) else lo
    pos = max(0, min(100, int((safe_temp - lo) / (hi - lo) * 100)))

    st.markdown(
        f"""
        <div style="position:relative; width:100%; height:14px; border-radius:7px;
                    background: linear-gradient(to right, #cce5ff 0%, #d4edda 40%, #fff3cd 65%, #f8d7da 100%);
                    margin-bottom:8px;">
          <div style="position:absolute; left:{pos}%; top:-6px; width:16px; height:16px;
                      background:black; border-radius:50%; transform: translateX(-50%);"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:13px; opacity:0.9;">
          <span>🧊 Calm</span><span>🌤 Active</span><span>🔥 Tense</span><span>🌪 Chaotic</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"Current temperature: **{temp:.3f} → {state}**" if not math.isnan(temp) else "Current temperature: NA")

    # ========= MOMENTUM PREVIEW =========
    st.markdown("### 2. ⏩ Momentum Preview (Early vs Now)")
    st.caption("🧠 This answers the question: **What changed so far?**.")
    lines = [
        bar_line("Early (0–48h):", temp_early),
        bar_line("Now:", temp_full),
        (f"{'Δ (Now−Early):':<14}  {delta:+.3f}" if not math.isnan(delta) else f"{'Δ (Now−Early):':<14}  NA"),
    ]
    st.code("\n".join(lines), language="text")

    # ========= Δ MOMENTUM RULER =========
    st.markdown("### 3. 📈 Momentum Shift (Δ after 48h)")
    st.caption("🧠 This is the **key signal** — Where is it heading?.")
    if math.isnan(delta):
        st.warning("Δ is NA for this video (early window missing).")
    else:
        d_lo, d_hi = -0.6, 0.6
        d = max(d_lo, min(d_hi, float(delta)))
        dpos = int((d - d_lo) / (d_hi - d_lo) * 100)

        st.markdown(
            f"""
            <div style="position:relative; width:100%; height:14px; border-radius:7px;
                        background: linear-gradient(to right,
                            #64b5f6 0%,
                            #e0e0e0 50%,
                            #ff8a65 100%
                        );
                        margin-bottom:8px;">
              <div style="position:absolute; left:{dpos}%; top:-6px; width:16px; height:16px;
                          background:black; border-radius:50%; transform: translateX(-50%);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:13px; opacity:0.9;">
              <span>📉 Cooling</span><span>➖ Steady</span><span>📈 Heating up</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"Δ (Now − Early): **{delta:+.3f}**   Left=cooling, center=steady, right=heating up.")

    # ========= CROWD LEANING (moved DOWN) =========
    st.markdown("### 4. 🗳️ Crowd Leaning (live)")
    st.caption("**What do others think?** This is **not money**, just vote share.")

    c1, c2, c3 = st.columns(3)
    c1.metric("YES share", f"{lean['p_yes']*100:.1f}%")
    c2.metric("NO share", f"{lean['p_no']*100:.1f}%")
    c3.metric("Votes", f"{lean['n']}")
    st.progress(lean["p_yes"])
    st.caption("This is vote share (not money). More votes = more stable signal.")

    # Simulator (debug)
    with st.expander("Debug: simulate 10 voters (to see how crowd leaning changes)", expanded=False):
        sim_n = st.slider("Simulated total voters", 1, 200, 10)
        sim_yes_pct = st.slider("Simulated YES share (%)", 0, 100, 60)
        sim = simulate_votes(sim_yes_pct / 100, sim_n)
        st.write(f"Simulated votes: YES={sim['n_yes']} / NO={sim['n_no']}")
        st.metric("Sim YES share", f"{sim['p_yes']*100:.1f}%")
        st.progress(sim["p_yes"])

    # ========= FINAL WHISTLE CLOCK =========
    st.markdown("### ⏱️ Final Whistle (when scoring opens)")
    if whistle_at is None:
        st.warning(f"Clock unavailable. Expected comments file: `{comments_path}`")
    else:
        is_over = pd.Timestamp.utcnow() >= whistle_at
        if is_over:
            st.success("⏹ Final whistle reached: scoring is available (admin).")
        else:
            st.markdown(
                f"""
                <div style="
                  display:flex; gap:16px; align-items:center;
                  padding:14px 16px; border-radius:12px;
                  background: rgba(0,0,0,0.03);
                  border: 1px solid rgba(0,0,0,0.06);
                ">
                  <div style="font-size:13px; opacity:0.7;">Time left</div>
                  <div style="font-size:32px; font-weight:900; letter-spacing:1px;">
                    {fmt_timedelta(remaining)}
                  </div>
                  <div style="font-size:13px; opacity:0.7;">
                    until Final Whistle
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

with right:
    st.subheader("🏆Foresight score")
    st.caption(
        """
        Your foresight score tracks how often your calls match what actually happens.
        After the **Final Whistle**:
        Correct call → **+1**
        Incorrect call → **−1**
        """
        )


    scores = load_scores(scores_mtime())
    st.caption("Higher score = stronger foresight over time.")

    me = scores[scores["user_id"].astype(str) == str(user_id)]
    if not me.empty:
        st.metric("Your score", f"{float(me.iloc[0]['score']):.0f}")
    else:
        st.metric("Your score", "0")

    with st.expander("Top foresight performers (after Final Whistle)"):
        if scores.empty:
            st.caption("No scores yet. Score a finished event to start this table.")
        else:
            st.dataframe(scores.head(15), use_container_width=True)

    st.divider()

    # Admin scoring collapsed
    with st.expander("Admin: score this event (Final Whistle)"):
        if whistle_at is None:
            st.caption("Scoring locked: needs comments file to compute Final Whistle time.")
        else:
            is_over = pd.Timestamp.utcnow() >= whistle_at
            if not is_over:
                st.info("Event still in play. Scoring opens after the Final Whistle.")
            else:
                st.success(f"Final whistle reached. Winner by v1 rule: **{winner}**")
                if st.button("Apply scoring now", use_container_width=True):
                    apply_final_whistle_scoring(str(selected_video), winner)
                    st.success("Scores updated.")
                    st.rerun()
# Footer (flush-left, no indent)
st.caption(
    f"Data: {FEATURES_PATH.name if FEATURES_PATH.exists() else FALLBACK_FEATURES_PATH.name}  "
    f"Meta CSV: {'ON' if META_PATH.exists() else 'OFF'}  "
    f"YouTube API: {'ON' if os.getenv('YOUTUBE_API_KEY','').strip() else 'OFF'}  "
    f"Votes: {VOTES_PATH.name}  Scores: {SCORES_PATH.name}"
)

