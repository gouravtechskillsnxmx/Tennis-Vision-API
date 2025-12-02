# filename: gpt_feedback.py
from typing import Dict, List
import os
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def generate_gpt_feedback(
    shot_type: str,
    features: Dict[str, float],
    strengths: List[str],
    improvements: List[str],
    score: int,
) -> str:
    """
    Returns a human-like coaching paragraph.
    """
    if client is None:
        # Fallback if no API key
        return (
            f"This is a {shot_type} with score {score}/100. "
            f"Strengths: {', '.join(strengths) or 'none'}. "
            f"Improvements: {', '.join(improvements) or 'none'}."
        )

    metrics_str = "\n".join(f"{k}: {v:.2f}" for k, v in features.items())

    prompt = f"""
You are a professional tennis coach analyzing a single shot.

Shot type: {shot_type}
Score: {score}/100

Raw metrics:
{metrics_str}

Detected strengths:
- {'\n- '.join(strengths) if strengths else 'None'}

Areas to improve:
- {'\n- '.join(improvements) if improvements else 'None'}

Write 1 short, friendly paragraph giving feedback to the player.
Use accessible language but keep it technically correct.
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert tennis coach."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()
