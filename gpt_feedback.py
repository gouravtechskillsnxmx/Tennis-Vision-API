# filename: gpt_feedback.py
from typing import Dict, List
		 
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

# Create client only if key exists
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def generate_gpt_feedback(
    shot_type: str,
    features: Dict[str, float],
    strengths: List[str],
    improvements: List[str],
    score: int,
) -> str:
    """
    Returns a human-like coaching paragraph for a single shot.
    Falls back to a simple text if no OPENAI_API_KEY is configured.
    """

    # Fallback: no OpenAI key, return simple text
    if client is None:
        strengths_txt = ", ".join(strengths) if strengths else "none"
        improvements_txt = ", ".join(improvements) if improvements else "none"
        return (
            f"This is a {shot_type} with a score of {score}/100. "
            f"Strengths: {strengths_txt}. "
            f"Areas to improve: {improvements_txt}."
        )

																		 

				 
															

					  
				  

			
			 

    # Build a readable metrics string
    metrics_lines = []
    for k, v in features.items():
        try:
            metrics_lines.append(f"{k}: {float(v):.2f}")
        except Exception:
            metrics_lines.append(f"{k}: {v}")
    metrics_str = "\n".join(metrics_lines) if metrics_lines else "No metrics available."

    # Build bullet lists for strengths and improvements
    strengths_str = (
        "\n".join(f"- {s}" for s in strengths) if strengths else "- None identified."
    )
    improvements_str = (
        "\n".join(f"- {s}" for s in improvements)
        if improvements
        else "- None identified."
    )

    # Plain string prompt (no triple-quoted f-strings to avoid syntax issues)
    prompt = (
        "You are a professional tennis coach analyzing a single shot.\n\n"
        f"Shot type: {shot_type}\n"
        f"Score: {score}/100\n\n"
        "Raw metrics:\n"
        f"{metrics_str}\n\n"
        "Detected strengths:\n"
        f"{strengths_str}\n\n"
        "Areas to improve:\n"
        f"{improvements_str}\n\n"
        "Write one short, friendly paragraph giving feedback to the player. "
        "Use accessible language but keep it technically correct and avoid repeating the bullet points verbatim."
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert tennis coach."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=220,
    )

    return response.choices[0].message.content.strip()
