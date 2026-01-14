from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import os
import json

from backend.model import load_model
from backend.predict import predict_image


# -----------------------------
# App init
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Paths (IMPORTANT FIX)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# -----------------------------
# Serve frontend static files
# -----------------------------
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# -----------------------------
# Home route (index.html)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

# -----------------------------
# Load classes & model
# -----------------------------
with open(BASE_DIR / "classes.json") as f:
    classes = json.load(f)

model = load_model(len(classes))

# -----------------------------
# Upload directory
# -----------------------------
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(model, str(file_path))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
