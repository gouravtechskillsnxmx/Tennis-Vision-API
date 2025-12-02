# filename: api_server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from main_pipeline import analyze_video

app = FastAPI()

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    tmp_dir = "uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}_{file.filename}")

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = analyze_video(tmp_path)
    return JSONResponse(result)

# Run with:
# uvicorn api_server:app --host 0.0.0.0 --port 8000
