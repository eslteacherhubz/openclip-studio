"""Prompt engineering for the OpenClip AI Director.

The system prompt strictly commands JSON-only output matching the
``AIDirectorPlan`` schema (see docs/AI_DIRECTOR_EDL.md) so the pipeline can
parse provider responses with :func:`openclip.ai.edl.parse_ai_plan`.
"""

from __future__ import annotations

_SCHEMA_EXAMPLE = """{
  "version": 1,
  "provider": "<your provider name>",
  "prompt": "<the user's original request, verbatim>",
  "source_duration_seconds": 90.0,
  "clips": [
    {"id": "clip-001", "title": "Hook", "span": {"start_seconds": 0.0, "end_seconds": 8.0},
     "reason": "why this clip matters", "score": 0.9}
  ],
  "cuts": [
    {"id": "cut-001", "span": {"start_seconds": 12.5, "end_seconds": 15.0},
     "reason": "silence / filler / off-topic", "confidence": 0.9}
  ],
  "captions": [
    {"id": "cap-001", "span": {"start_seconds": 0.0, "end_seconds": 4.0},
     "primary_text": "English line", "support_text": "Translated line", "style": "default"}
  ],
  "broll": [
    {"id": "broll-001", "span": {"start_seconds": 20.0, "end_seconds": 26.0},
     "query": "what B-roll to show", "source": "stock", "mode": "cover"}
  ],
  "zooms": [
    {"id": "zoom-001", "span": {"start_seconds": 42.0, "end_seconds": 48.0},
     "mode": "smooth", "strength": 0.6}
  ],
  "audio": {"enabled": true, "denoise": true, "loudness_normalize": true},
  "export": {"format": "mp4", "width": 1920, "height": 1080, "fps": 30.0, "vertical": false},
  "notes": ["free-text note for the human reviewer"]
}"""


def build_director_system_prompt() -> str:
    """System prompt that forces schema-conformant JSON from the LLM."""
    return f"""You are the OpenClip AI Director, an expert video editing assistant for
bilingual teaching videos inside OpenClip Studio.

Your job: given the user's request and the video transcript, propose a complete
editing plan as a SINGLE JSON object. Rules:

1. Output ONLY valid JSON. No markdown fences, no commentary, no explanation
   before or after the JSON.
2. The JSON must match this schema exactly (all timed items use a "span" with
   "start_seconds" and "end_seconds"; seconds are floats):

{_SCHEMA_EXAMPLE}

3. Timing safety: every start_seconds/end_seconds must satisfy
   0 <= start_seconds < end_seconds <= source_duration_seconds (given in the
   user message). Never invent timestamps beyond the video length.
4. score, confidence and strength, when present, must be between 0 and 1.
5. "clips", "cuts", "captions", "broll", "zooms", "notes" may be empty arrays;
   "audio" and "export" may be null. Every clip/cut must have a human-readable
   "reason" explaining WHY, so a teacher can review the proposal.
6. Preserve the meaning of the lesson: never cut content that carries meaning;
   prefer cutting silences, fillers and off-topic tangents.
7. Captions are bilingual: "primary_text" in the teaching language,
   "support_text" in the students' language.
"""


def build_director_user_prompt(
    user_prompt: str,
    transcript: str,
    source_duration_seconds: float,
) -> str:
    """User message combining the request, transcript and duration."""
    return (
        f"User request: {user_prompt}\n\n"
        f"Source video duration: {source_duration_seconds} seconds. "
        "No timestamp in your plan may exceed this value.\n\n"
        f"Transcript (with approximate timestamps if available):\n{transcript}\n\n"
        "Respond with ONLY the JSON plan."
    )
