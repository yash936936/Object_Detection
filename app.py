from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import io
import os
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Object Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
logger.info("📂 Static directory initialized")

# Copy index.html to static directory
try:
    with open("index.html", "r", encoding="utf-8") as source:
        with open("static/index.html", "w", encoding="utf-8") as f:
            f.write(source.read())
    logger.info("📄 Copied index.html to static directory")
except FileNotFoundError:
    logger.error("❌ index.html not found in project directory")
    raise FileNotFoundError("index.html not found. Please ensure it exists in the project directory.")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model path
MODEL_PATH = "best.pt"

# Initialize YOLO model
model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    model = None

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse('static/index.html')

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Object detection endpoint"""
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    # Validate file type
    if not file.content_type.startswith('image/'):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, JPEG, WebP)")

    try:
        # Read and process image
        contents = await file.read()
        logger.info(f"📦 File received: {file.filename}, Size: {len(contents)} bytes")

        # Open image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        logger.info(f"🖼️ Image size: {image.size}")

        # Run detection
        results = model(image, conf=0.25, iou=0.45)  # Added IoU threshold for better NMS
        logger.info("✅ Detection completed")

        # Process results
        predictions = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = [float(coord) for coord in box.xywh[0]]

                    prediction = {
                        "class_name": model.names[cls_id],
                        "confidence": confidence,
                        "bbox": bbox  # [center_x, center_y, width, height]
                    }
                    predictions.append(prediction)

        logger.info(f"📤 Returning {len(predictions)} predictions")
        return {"predictions": predictions, "status": "success"}

    except Exception as e:
        logger.error(f"❌ Error during detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_classes": len(model.names) if model else 0
    }

if __name__ == "__main__":
    # Run the server with import string to enable reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)