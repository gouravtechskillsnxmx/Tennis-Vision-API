# api_server.py
import os
import uuid
import shutil
import logging
import traceback

logger = logging.getLogger("tennis-vision-api")
logging.basicConfig(level=logging.INFO)

from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from redis import Redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

# ---------- Logging ----------
logger = logging.getLogger("tennis-vision-api")
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "tennis")
DATA_DIR = os.getenv("DATA_DIR", "/data")

# Directories for videos/results/overlays
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
RESULT_DIR = os.path.join(DATA_DIR, "results")
OVERLAY_DIR = os.path.join(DATA_DIR, "overlays")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# ---------- Redis / RQ ----------
redis_conn = Redis.from_url(REDIS_URL)
job_queue = Queue(RQ_QUEUE_NAME, connection=redis_conn)

# ---------- FastAPI app ----------
app = FastAPI(title="Tennis Vision API", version="1.0.0")

# api_server.py

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
async def health():
    return {"status": "ok"}

def run_analysis(video_path: str):
    # IMPORTANT: import inside function so startup is fast
    from main_pipeline import analyze_video
    return analyze_video(video_path)

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info("Saved upload to %s", tmp_path)

        result = run_analysis(tmp_path)
        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Error in /analyze: %s", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "type_of_shot": "error",
                "strengths": [],
                "improvements": [f"Server error: {str(e)}"],
                "score": 0,
                "overlay_url": "",
            },
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        # optional cleanup:
        # try: os.remove(tmp_path)
        # except: pass


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Check status (and result if finished) for a given job_id.
    - Uses RQ Job as the source of truth.
    """
    try:
        job: Job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()  # queued, started, finished, failed, deferred

    response = {
        "job_id": job.id,
        "status": status,
    }

    if status == "finished":
        # process_video_job returns a dict -> job.result
        response["result"] = job.result
    elif status == "failed":
        # You can expose less/more here; keeping it simple
        response["error"] = "Job failed. Check server logs for details."

    return JSONResponse(response)
