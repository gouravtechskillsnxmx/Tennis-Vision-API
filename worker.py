# worker.py
import os
import logging

from redis import Redis
from rq import Worker, Queue, Connection

from main_pipeline import analyze_video  # your existing pipeline

# ---------- Logging ----------
logger = logging.getLogger("tennis-vision-worker")
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "tennis")
DATA_DIR = os.getenv("DATA_DIR", "/data")

VIDEO_DIR = os.path.join(DATA_DIR, "videos")
RESULT_DIR = os.path.join(DATA_DIR, "results")
OVERLAY_DIR = os.path.join(DATA_DIR, "overlays")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)


def process_video_job(job_id: str) -> dict:
    """
    Background job function.

    - Loads the saved video file for this job_id.
    - Runs the full tennis analysis pipeline (analyze_video).
    - Returns the result dict (stored in RQ job.result).
    """
    video_path = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
    logger.info("Worker processing job_id=%s, video=%s", job_id, video_path)

    if not os.path.exists(video_path):
        msg = f"Video file not found for job {job_id}: {video_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Call your existing pipeline (already developed)
    result = analyze_video(video_path)

    # If analyze_video creates an overlay video and returns a relative path,
    # you can standardize overlay_url here, e.g.:
    # result["overlay_url"] = f"https://YOUR_DOMAIN/overlays/{job_id}.mp4"
    # (Assuming you later add a /overlays/{job_id}.mp4 endpoint.)

    logger.info("Job %s completed successfully", job_id)
    return result


if __name__ == "__main__":
    logger.info("Starting RQ worker. REDIS_URL=%s, QUEUE=%s", REDIS_URL, RQ_QUEUE_NAME)
    redis_conn = Redis.from_url(REDIS_URL)
    with Connection(redis_conn):
        worker = Worker([RQ_QUEUE_NAME])
        worker.work()
