# api_server.py — PURE SYNCHRONOUS (Option 1)

import os
import uuid
import shutil
import logging
import traceback

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tennis-vision-api")

app = FastAPI(title="Tennis Vision API", version="1.0.0")

UPLOAD_DIR = "/tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


def run_analysis(video_path: str):
    # IMPORTANT: lazy import (prevents slow startup / crashes)
    from main_pipeline import analyze_video
    return analyze_video(video_path)


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")

    try:
        # Save upload
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Saved upload to {tmp_path}")

        # Run pipeline synchronously
        result = run_analysis(tmp_path)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Analyze failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "type_of_shot": "error",
                "strengths": [],
                "improvements": [str(e)],
                "score": 0,
                "overlay_url": "",
            },
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass
