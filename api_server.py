# api_server.py
import os, uuid, shutil, logging, traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("tennis-vision-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Tennis Vision API")


@app.get("/health")
async def health():
    return {"status": "ok"}


def run_analysis(video_path: str):
    from main_pipeline import analyze_video  # or main_pipeline_minimal if you’re still debugging
    return analyze_video(video_path)


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    tmp_dir = "uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}_{file.filename}")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info("Saved upload to %s (%s bytes)", tmp_path, file.size or "unknown")
        result = run_analysis(tmp_path)
        logger.info("Analysis done")
        return JSONResponse(result)

    except Exception as e:
        logger.error("Error during analysis: %s", e)
        traceback.print_exc()
        # Return JSON instead of crashing -> avoids Render 502
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
