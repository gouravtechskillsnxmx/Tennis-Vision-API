from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, uuid, shutil, logging, traceback

app = FastAPI(title="Tennis Vision API")

@app.get("/health")
async def health():
    return {"status": "ok"}


def run_analysis(video_path: str):
    """
    Lazy import so heavy modules are only loaded when needed.
    For now, we use the minimal pipeline to debug stability.
    """
    from main_pipeline_minimal import analyze_video
    return analyze_video(video_path)


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    tmp_dir = "uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = run_analysis(tmp_path)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
