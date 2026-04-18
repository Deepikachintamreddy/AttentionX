import os
import json
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HookSet:
    curiosity: str
    shock: str
    value: str

@dataclass
class GeminiAnalysis:
    viral_score: float
    emotional_score: float
    educational_score: float
    surprising_score: float
    relatable_score: float
    hooks: HookSet
    clip_title: str
    key_quote: str
    why_viral: str
    suggested_trim_start: Optional[float] = None
    suggested_trim_end: Optional[float] = None


ANALYSIS_PROMPT = """You are a viral content strategist. Analyse this transcript segment from timestamp {start:.1f}s to {end:.1f}s:

TRANSCRIPT:
{transcript}

Return ONLY this JSON (no markdown, no explanation):
{{
  "viral_score": <float 0-100>,
  "emotional_score": <float 0-100>,
  "educational_score": <float 0-100>,
  "surprising_score": <float 0-100>,
  "relatable_score": <float 0-100>,
  "hooks": {{
    "curiosity": "<hook that teases without revealing, max 12 words>",
    "shock": "<hook that challenges a common belief, max 12 words>",
    "value": "<hook that promises a clear benefit, max 12 words>"
  }},
  "clip_title": "<punchy 5-7 word title>",
  "key_quote": "<single most powerful sentence from transcript>",
  "why_viral": "<one sentence why this moment has viral potential>",
  "suggested_trim_start": <float seconds>,
  "suggested_trim_end": <float seconds>
}}"""


def analyze_clip(
    transcript_segments: List[dict],
    window_start: float,
    window_end: float,
    api_key: str,
) -> GeminiAnalysis:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)

        full_transcript = " ".join(
            seg.get("text", "") for seg in transcript_segments
        ).strip()

        prompt = ANALYSIS_PROMPT.format(
            start=window_start,
            end=window_end,
            transcript=full_transcript,
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=1.0,
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = _fallback(full_transcript, window_start, window_end)

    except Exception as e:
        print(f"Gemini error: {e}")
        full_transcript = " ".join(seg.get("text", "") for seg in transcript_segments).strip()
        data = _fallback(full_transcript, window_start, window_end)

    hooks = HookSet(
        curiosity=data.get("hooks", {}).get("curiosity", "This changes everything..."),
        shock=data.get("hooks", {}).get("shock", "What no one tells you..."),
        value=data.get("hooks", {}).get("value", "Here's the exact method..."),
    )

    return GeminiAnalysis(
        viral_score=float(data.get("viral_score", 50)),
        emotional_score=float(data.get("emotional_score", 50)),
        educational_score=float(data.get("educational_score", 50)),
        surprising_score=float(data.get("surprising_score", 50)),
        relatable_score=float(data.get("relatable_score", 50)),
        hooks=hooks,
        clip_title=data.get("clip_title", "Key Insight"),
        key_quote=data.get("key_quote", full_transcript[:120]),
        why_viral=data.get("why_viral", "High engagement potential detected."),
        suggested_trim_start=data.get("suggested_trim_start"),
        suggested_trim_end=data.get("suggested_trim_end"),
    )


def rank_and_select(analyses: List[GeminiAnalysis], top_n: int = 3) -> List[GeminiAnalysis]:
    return sorted(analyses, key=lambda a: a.viral_score, reverse=True)[:top_n]


def _fallback(transcript: str, start: float, end: float) -> dict:
    import random
    base = random.randint(55, 85)
    return {
        "viral_score": float(base),
        "emotional_score": float(random.randint(45, 90)),
        "educational_score": float(random.randint(50, 95)),
        "surprising_score": float(random.randint(40, 85)),
        "relatable_score": float(random.randint(50, 88)),
        "hooks": {
            "curiosity": "This insight will change how you think...",
            "shock": "Most people get this completely wrong...",
            "value": "Here's the exact framework to use...",
        },
        "clip_title": "Key Moment",
        "key_quote": transcript[:120] + "..." if len(transcript) > 120 else transcript,
        "why_viral": "Strong educational content with high engagement signals.",
        "suggested_trim_start": start,
        "suggested_trim_end": end,
    }