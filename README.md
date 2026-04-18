# ⚡ AttentionX — Automated Content Repurposing Engine

> Turn a 60-minute lecture into 3 viral-ready short clips in minutes.

[![Demo Video](https://img.shields.io/badge/Demo-Watch%20Now-f7c948?style=for-the-badge&logo=youtube)](YOUR_DEMO_LINK_HERE)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/DeepikaChintamreddy/AttentionX)

---

## 🎯 The Problem

Mentors and educators produce hours of high-value long-form content. Modern audiences consume information in 60-second bursts. Valuable insights are buried — inaccessible, unwatched.

**AttentionX solves this automatically.**

---

## ✨ What Makes AttentionX Different

Most tools just cut clips. AttentionX **understands** your content:

| Feature | Others | AttentionX |
|---|---|---|
| Clip detection | Simple silence detection | Audio energy + speech-rate combined **AttentionScore** |
| AI scoring | Basic summarization | 4-axis Viral Score (Emotional, Educational, Surprising, Relatable) |
| Hook writing | One generic headline | **3 hook styles** per clip: Curiosity / Shock / Value |
| Captions | Static subtitles | **Karaoke-style** word-level highlights |
| Face tracking | None | MediaPipe with **exponential smoothing** (no jitter) |
| UI | Upload → download | Live **Attention Timeline** + Score Dashboard |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                   │
│  Upload → Timeline Chart → Score Cards → Preview     │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                Pipeline Orchestrator                  │
└──┬──────────────┬──────────────┬────────────────┬───┘
   │              │              │                │
   ▼              ▼              ▼                ▼
Audio          Whisper        Gemini          Video
Analyzer      Transcriber    Analyzer        Processor
(Librosa)    (Word-level)   (Flash 1.5)    (MediaPipe
 RMS + ZCR    timestamps    Viral scoring  + MoviePy)
```

### Core Modules

| Module | Purpose |
|---|---|
| `core/audio_analyzer.py` | Librosa RMS + ZCR → AttentionScore → peak clustering |
| `core/transcriber.py` | faster-whisper with word timestamps → karaoke captions |
| `core/gemini_analyzer.py` | Gemini 1.5 Flash → 4-axis viral scoring + 3 hook styles |
| `core/video_processor.py` | MediaPipe face tracking + MoviePy crop + ffmpeg caption burn |
| `core/pipeline.py` | Orchestration with live progress callbacks |
| `app.py` | Streamlit UI with timeline, score cards, download |

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/attentionx
cd attentionx
pip install -r requirements.txt
```

### 2. Get a free Gemini API key
Visit [Google AI Studio](https://aistudio.google.com/) — free tier is sufficient.

### 3. Run
```bash
streamlit run app.py
```

Enter your Gemini API key in the sidebar, upload a video, click **Generate Clips**.

---

## 🧠 How AttentionScore Works

AttentionScore combines two audio signals per frame:

```
AttentionScore = 0.65 × RMS_energy_normalized
              + 0.35 × ZeroCrossingRate_normalized
```

- **RMS energy** → captures passion, volume, emotional intensity
- **Zero-crossing rate** → captures rapid speech, excitement bursts
- **3-second smoothing** → removes noise, keeps real peaks
- **Minimum 60s gap enforcement** → ensures diverse, non-overlapping clips

---

## 🎬 Viral Score Card — 4 Dimensions

Gemini 1.5 Flash scores each transcript segment on:

- **Emotional (0–100)**: How emotionally charged is this moment?
- **Educational (0–100)**: How much practical value does it contain?
- **Surprising (0–100)**: How counterintuitive or unexpected?
- **Relatable (0–100)**: How universally relatable is the experience?

The composite **Viral Score** ranks clips for final selection.

---

## 📱 Output Format

- Resolution: **1080×1920** (9:16 vertical — TikTok/Reels/Shorts ready)
- Captions: Word-level karaoke highlights (yellow active word)
- Hook overlay: First 2.5 seconds with title + selected hook style
- Duration: Configurable 30–90 seconds (default 60s)

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Plotly |
| AI/NLP | Google Gemini 1.5 Flash |
| Transcription | faster-whisper / OpenAI Whisper |
| Audio | Librosa |
| Computer Vision | MediaPipe Face Detection |
| Video | MoviePy + ffmpeg |
| Backend | Python 3.10+ |

---

## 📊 Evaluation Criteria Alignment

| Criterion | How AttentionX addresses it |
|---|---|
| **Impact (20%)** | Fully automated end-to-end pipeline; one upload produces ready-to-post clips |
| **Innovation (20%)** | Combined AttentionScore signal; 3-style hook engine; 4-axis viral scoring |
| **Technical (20%)** | Clean modular architecture; graceful fallbacks; face-smoothing algorithm |
| **UX (25%)** | Dark-mode dashboard; live timeline; score cards; one-click download |
| **Presentation (15%)** | See demo video link at top |

---

## 🎥 Demo

📽️ **[Watch the full demo here](https://drive.google.com/file/d/1aJXB3y76dmwOoi6q3wlkqdpYlzC-AM8s/view?usp=sharing)**

The demo shows:
1. Uploading a 30-minute lecture
2. Live attention timeline rendering
3. 3 viral score cards with hook options
4. Final vertical clips with karaoke captions

---

## 👤 Author

Built for the **AttentionX AI Hackathon** by UnsaidTalks Education.
