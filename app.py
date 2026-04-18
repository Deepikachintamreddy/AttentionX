import streamlit as st
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="AttentionX — Viral Clip Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0a0f; }
  [data-testid="stSidebar"] { background: #111118; border-right: 1px solid #333; }
  [data-testid="stSidebar"] label { color: #d0d0e8 !important; font-size: 0.92rem; }
  [data-testid="stSidebar"] p     { color: #b0b0cc !important; }
  [data-testid="stSidebar"] span  { color: #d0d0e8 !important; }
  [data-testid="stSidebar"] h3    { color: #f7c948 !important; }
  [data-testid="stSidebar"] [data-testid="stTickBarMin"],
  [data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #666 !important; }
  [data-testid="stSidebar"] .stRadio > div > label > div > p { color: #c0c0d8 !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] span { color: #d0d0e8 !important; }
  [data-testid="stFileUploadDropzone"] {
    border: 1px dashed #333 !important;
    border-radius: 8px !important;
    background: #0e0e1c !important;
  }
  [data-testid="stFileUploadDropzone"]:hover {
    border-color: #f7c948 !important;
    background: #111120 !important;
  }
  [data-testid="stFileUploadDropzone"] span { color: #aaa !important; }
  [data-testid="stFileUploadDropzone"] p    { color: #888 !important; }
  h1, h2, h3 { color: #f0f0ff; }
  .score-card {
    background: linear-gradient(135deg, #13131e, #1a1a2e);
    border: 1px solid #2a2a4a; border-radius: 12px;
    padding: 20px; margin-bottom: 16px;
  }
  .viral-score-big {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #f7c948, #ff6b35);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hook-pill {
    display: inline-block; background: #1e1e3a;
    border: 1px solid #3a3a6a; border-radius: 20px;
    padding: 5px 14px; margin: 3px 3px 3px 0;
    font-size: 0.83rem; color: #aab;
  }
  .hook-selected {
    background: #f7c948 !important; color: #111 !important;
    border-color: #f7c948 !important; font-weight: 600;
  }
  .metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
  .metric-box {
    flex: 1; min-width: 88px; background: #0e0e1c;
    border-radius: 8px; padding: 10px 12px;
    border: 1px solid #222240; text-align: center;
  }
  .metric-label { font-size: 0.67rem; color: #666; text-transform: uppercase; letter-spacing: .08em; }
  .metric-val   { font-size: 1.3rem; font-weight: 700; color: #f7c948; }
  .key-quote {
    border-left: 3px solid #f7c948; padding-left: 14px;
    color: #ccc; font-style: italic; margin: 10px 0; font-size: 0.91rem;
  }
  .why-viral { color: #7aeba0; font-size: 0.87rem; margin-top: 8px; }
  .rank-badge {
    background: #f7c948; color: #111; border-radius: 50%;
    width: 30px; height: 30px; display: inline-flex;
    align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.9rem; margin-right: 8px;
  }
  .stProgress > div > div > div {
    background: linear-gradient(90deg, #f7c948, #ff6b35) !important;
  }
  .stButton > button {
    background: linear-gradient(90deg, #f7c948, #ff6b35);
    color: #111 !important; font-weight: 700; border: none;
    border-radius: 8px; padding: 0.6rem 2rem; font-size: 1rem;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(247,201,72,0.4);
  }
  .log-box {
    background: #080810; border: 1px solid #1a1a2a; border-radius: 6px;
    padding: 10px; font-size: 0.8rem; color: #4fc98a;
    font-family: monospace; max-height: 130px; overflow: auto;
  }
</style>
""", unsafe_allow_html=True)


def _fmt(sec: float) -> str:
    return f"{int(sec//60)}:{int(sec%60):02d}"


def _run_analysis(uploaded, gemini_key, clip_duration, top_n_peaks,
                  top_n_clips, whisper_model, hook_type):
    from core.pipeline import run_pipeline

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    output_dir = tempfile.mkdtemp()

    st.markdown("---")
    st.markdown("### 🔄 Processing")
    stage_text   = st.empty()
    progress_bar = st.progress(0)
    log_box      = st.empty()
    log_lines    = []

    def on_progress(stage: str, pct: float):
        stage_text.markdown(
            f"<p style='color:#f7c948;font-weight:600;margin:4px 0;'>▶ {stage}</p>",
            unsafe_allow_html=True
        )
        progress_bar.progress(min(pct, 1.0))
        log_lines.append(f"✓ {stage}")
        log_box.markdown(
            "<div class='log-box'>" + "<br>".join(log_lines[-6:]) + "</div>",
            unsafe_allow_html=True
        )

    try:
        result = run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            gemini_api_key=gemini_key,
            clip_duration=clip_duration,
            top_n_peaks=top_n_peaks,
            top_n_clips=top_n_clips,
            whisper_model=whisper_model,
            hook_type=hook_type,
            on_progress=on_progress,
        )
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

    progress_bar.progress(1.0)
    stage_text.markdown(
        "<p style='color:#4fc98a;font-weight:700;font-size:1.05rem;'>✅ All clips ready!</p>",
        unsafe_allow_html=True
    )

    # Attention Timeline
    st.markdown("---")
    st.markdown("### 📈 Attention Timeline")
    st.markdown(
        "<p style='color:#555;font-size:0.87rem;margin-top:-8px;'>"
        "Full-video audio energy · orange markers = selected clip peaks</p>",
        unsafe_allow_html=True
    )

    if result.timeline_times and result.timeline_scores:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.timeline_times,
            y=result.timeline_scores,
            fill="tozeroy",
            line=dict(color="#f7c948", width=1.5),
            fillcolor="rgba(247,201,72,0.09)",
            hovertemplate="%{x:.1f}s — Score: %{y:.1f}<extra></extra>",
        ))
        for clip in result.clips:
            fig.add_vline(
                x=clip.peak.peak_sec,
                line_dash="dash", line_color="#ff6b35",
                annotation_text=f"  #{clip.rank}",
                annotation_font_color="#ff6b35",
                annotation_font_size=13,
            )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,20,1)",
            font_color="#888",
            xaxis=dict(title="Time (seconds)", gridcolor="#1a1a2e", color="#666"),
            yaxis=dict(title="Attention Score", gridcolor="#1a1a2e", color="#666", range=[0, 108]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=210, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Clip cards
    st.markdown("---")
    st.markdown(f"### 🎬 Your Top {len(result.clips)} Viral Clips")

    for clip in result.clips:
        a, p = clip.analysis, clip.peak

        hook_pills = ""
        for style, icon, text in [
            ("curiosity", "🎯", a.hooks.curiosity),
            ("shock",     "⚡", a.hooks.shock),
            ("value",     "💡", a.hooks.value),
        ]:
            cls = "hook-pill hook-selected" if hook_type == style else "hook-pill"
            hook_pills += f"<span class='{cls}'>{icon} {text}</span><br>"

        st.markdown(f"""
        <div class='score-card'>
          <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap;'>
            <span class='rank-badge'>{clip.rank}</span>
            <span style='color:#f0f0ff;font-size:1.02rem;font-weight:600;'>{a.clip_title}</span>
            <span class='viral-score-big' style='margin-left:auto;'>{a.viral_score:.0f}</span>
            <span style='color:#444;font-size:0.73rem;'>/ 100</span>
          </div>
          <div class='metric-row'>
            <div class='metric-box'><div class='metric-label'>Emotional</div><div class='metric-val'>{a.emotional_score:.0f}</div></div>
            <div class='metric-box'><div class='metric-label'>Educational</div><div class='metric-val'>{a.educational_score:.0f}</div></div>
            <div class='metric-box'><div class='metric-label'>Surprising</div><div class='metric-val'>{a.surprising_score:.0f}</div></div>
            <div class='metric-box'><div class='metric-label'>Relatable</div><div class='metric-val'>{a.relatable_score:.0f}</div></div>
            <div class='metric-box'><div class='metric-label'>Audio Peak</div><div class='metric-val'>{p.attention_score:.0f}</div></div>
            <div class='metric-box'><div class='metric-label'>Timestamp</div><div class='metric-val' style='font-size:0.88rem;'>{_fmt(p.start_sec)}</div></div>
          </div>
          <div class='key-quote'>"{a.key_quote}"</div>
          <div class='why-viral'>💡 {a.why_viral}</div>
          <div style='margin-top:14px;'>
            <div style='color:#444;font-size:0.68rem;margin-bottom:5px;
                        text-transform:uppercase;letter-spacing:.08em;'>Hook options</div>
            {hook_pills}
          </div>
        </div>
        """, unsafe_allow_html=True)

        if os.path.exists(clip.output_path):
            cv, cd = st.columns([3, 1])
            with cv:
                with open(clip.output_path, "rb") as f:
                    st.video(f.read())
            with cd:
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                with open(clip.output_path, "rb") as f:
                    st.download_button(
                        f"⬇️ Download #{clip.rank}",
                        data=f.read(),
                        file_name=f"attentionx_clip_{clip.rank}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                st.markdown(f"""
                <div style='font-size:0.74rem;color:#444;text-align:center;
                            margin-top:6px;line-height:1.6;'>
                  {_fmt(p.start_sec)} → {_fmt(p.end_sec)}<br>
                  {p.duration:.0f}s · 1080×1920
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱ Total time",       f"{result.total_time_sec:.0f}s")
    c2.metric("🎬 Clips produced",   len(result.clips))
    if result.clips:
        c3.metric("⚡ Top viral score", f"{result.clips[0].analysis.viral_score:.0f}/100")
        avg = sum(c.analysis.viral_score for c in result.clips) / len(result.clips)
        c4.metric("📊 Avg score",       f"{avg:.0f}/100")


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 6px;'>
      <span style='font-size:1.4rem;font-weight:800;color:#f7c948;'>⚡ AttentionX</span>
      <span style='font-size:0.7rem;color:#555;margin-left:6px;'>AI ENGINE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    import os
    gemini_key = st.text_input(
        "🔑 Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        placeholder="AIza...",
        help="Free from aistudio.google.com"
    )

    st.markdown("---")
    st.markdown("**🎬 Clip Settings**")
    clip_duration = st.slider("Clip length (seconds)", 30, 90, 30, 5)
    top_n_peaks   = st.slider("Peak candidates to scan", 3, 8, 3)
    top_n_clips   = st.slider("Final clips to produce", 1, 5, 1)

    st.markdown("---")
    st.markdown("**🤖 AI Settings**")
    whisper_model = st.selectbox(
        "Whisper model speed",
        ["tiny", "base", "small"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**🎯 Hook Style**")
    hook_type = st.radio(
        "Choose hook framing",
        ["curiosity", "shock", "value"],
        format_func=lambda x: {
            "curiosity": "🎯 Curiosity hook",
            "shock":     "⚡ Shock hook",
            "value":     "💡 Value hook",
        }[x],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#555;line-height:1.7;'>
      Built for AttentionX Hackathon<br>
      Gemini 1.5 Flash · Whisper · MediaPipe<br>
      <span style='color:#f7c948;'>Max upload: 2 GB</span>
    </div>
    """, unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:2px;'>⚡ AttentionX</h1>
<p style='color:#666;font-size:1rem;margin-top:0;'>
  Upload a long-form video → get viral-ready short clips in minutes
</p>
""", unsafe_allow_html=True)
st.markdown("---")

st.markdown(
    "#### 📤 Upload your video "
    "<span style='color:#555;font-size:0.8rem;font-weight:400;'>"
    "MP4 · MOV · AVI · MKV &nbsp;|&nbsp; up to 2 GB</span>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Drag and drop or browse",
    type=["mp4", "mov", "avi", "mkv"],
    label_visibility="collapsed",
)

if uploaded:
    size_mb = uploaded.size / (1024 * 1024)
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(f"""
        <div style='background:#0e0e1c;border:1px solid #2a2a4a;border-radius:8px;
                    padding:12px 18px;display:flex;align-items:center;gap:12px;'>
          <span style='font-size:1.4rem;'>📹</span>
          <div>
            <div style='color:#f0f0ff;font-weight:600;font-size:0.95rem;'>{uploaded.name}</div>
            <div style='color:#555;font-size:0.8rem;'>{size_mb:.1f} MB</div>
          </div>
          <span style='margin-left:auto;background:#0d1f0d;color:#4fc98a;
                       padding:3px 12px;border-radius:12px;font-size:0.76rem;
                       border:1px solid #1a3a1a;'>✓ Ready</span>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        run_btn = st.button("🚀 Generate Clips", use_container_width=True)

    if not gemini_key:
        st.warning("⚠️  Add your Gemini API key in the sidebar to begin.")
    elif run_btn:
        _run_analysis(uploaded, gemini_key, clip_duration, top_n_peaks,
                      top_n_clips, whisper_model, hook_type)

else:
    st.markdown("""
    <div style='text-align:center;padding:48px 20px 28px;'>
      <div style='font-size:3.2rem;'>🎬</div>
      <h2 style='color:#888;margin:12px 0 8px;'>Drop a video to get started</h2>
      <p style='color:#666;max-width:460px;margin:0 auto;font-size:0.92rem;line-height:1.6;'>
        AttentionX finds the highest-energy moments, scores them for viral potential,
        and produces vertical clips with hook overlays — automatically.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🎵", "Attention Timeline",
         "Full-video audio energy chart with peak markers."),
        (c2, "🧠", "Viral Score Cards",
         "4-axis AI scoring: Emotional · Educational · Surprising · Relatable."),
        (c3, "✍️", "3-Style Hook Engine",
         "Curiosity, Shock or Value hooks — Gemini writes 3 options per clip."),
    ]:
        with col:
            st.markdown(f"""
            <div class='score-card'>
              <div style='font-size:1.7rem;margin-bottom:8px;'>{icon}</div>
              <h3 style='color:#e0e0ff;margin:0 0 6px;font-size:0.97rem;'>{title}</h3>
              <p style='color:#777;font-size:0.83rem;margin:0;line-height:1.55;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)