# api_server.py
import os
import uuid
import shutil
import logging
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


@app.get("/health")
async def health():
    """
    Simple health endpoint so you (and Render) can verify the service is alive.
    """
    return {"status": "ok"}


@app.post("/analyze")
async def enqueue_analysis(file: UploadFile = File(...)):
    """
    Enqueue a video for background analysis.
    - Saves the uploaded file to DATA_DIR/videos/{job_id}.mp4
    - Enqueues an RQ job that runs worker.process_video_job(job_id)
    - Returns job_id and initial status
    """
    # Generate job_id (also used as RQ job id)
    job_id = uuid.uuid4().hex

    # Save video to disk
    video_path = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Saved uploaded video to %s (job_id=%s)", video_path, job_id)
    except Exception as e:
        logger.exception("Failed to save uploaded file: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Enqueue background job: worker.process_video_job(job_id)
    try:
        rq_job = job_queue.enqueue(
            "worker.process_video_job",  # string: module.func
            job_id,                      # arg passed to process_video_job
            job_id=job_id,               # force RQ job id = our job_id
        )
        logger.info("Enqueued job %s on queue %s", rq_job.id, RQ_QUEUE_NAME)
    except Exception as e:
        logger.exception("Failed to enqueue job: %s", e)
        raise HTTPException(status_code=500, detail="Failed to enqueue analysis job")

    # Return lightweight response immediately
    return {
        "job_id": rq_job.id,
        "status": rq_job.get_status(),  # 'queued'
        "message": "Video received; analysis will run in background.",
    }


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
