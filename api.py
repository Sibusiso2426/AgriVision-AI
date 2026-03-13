from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import uvicorn
from vision import AgriVisionModel
from ultralytics import YOLO
import cv2
import numpy as np
import os

try:
    from vision import AgriVisionModel
except ModuleNotFoundError:
    from vision import AgriVisionModel

app = FastAPI(title="AgriVision AI Multi-Modal API")

@app.get("/")
def home():
    return {"status": "AgriVision Backend is Online"}

# 1. Initialize Vision Engines
vision_engine = AgriVisionModel()
# Load YOLOv8; ensure the .pt file is in your root or models folder
# Use the official YOLOv8 nano model you already have
livestock_model = YOLO('yolov8n.pt')

# 1. The "Knowledge Base" Function
def get_recommendation(label):
    recommendations = {
        "Early Blight": "💡 Advice: Improve air circulation and remove infected lower leaves.",
        "Late Blight": "⚠️ Warning: Highly contagious! Remove infected plants immediately.",
        "Leaf Mold": "💡 Advice: Reduce humidity and improve spacing between plants.",
        "Healthy": "✅ Great news! Your crop looks healthy.",
    }
    # We use 'label' here to find the matching advice
    return recommendations.get(label, "Monitor crop and consult a local officer.")

# 2. The API Route
@app.post("/api/v1/plant/detect")
async def detect_plant_disease(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Get results from your class
    label, confidence = vision_engine.predict_plant(contents)
    
    # We call the function and save it to a variable called 'final_advice'
    final_advice = get_recommendation(label)
    
    return {
        "status": "success",
        "label": label,
        "confidence": f"{confidence * 100:.2f}%",
        "recommendation": final_advice  # <--- Make sure this matches the variable name above!
    }
# --- LIVESTOCK ROUTES ---
@app.post("/api/v1/livestock/detect")
async def analyze_livestock(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image data"})

    results = livestock_model(img)
    print(f"DEBUG: Model results type is: {type(results)}")
    if results is None:
        return {"detections": [], "alert_triggered": False, "count": 0}
# -----------------------------
    detections = []
    alert_triggered = False
    
    # List of classes your custom model would identify as sick
    DISEASE_CLASSES = ["Lumpy Skin", "Lesion", "Mouth Ulcer", "Injury"]

    for r in results:
        for box in r.boxes:
            label = livestock_model.names[int(box.cls[0].item())]
            conf = box.conf[0].item()
            
# SIMULATION LOGIC: 
            # If it's a cow and confidence is > 85%, we'll flag it for a health check
            if label == "cow" and conf > 0.85:
                alert_triggered = True
                label = "Cow (Health Alert: Check Skin)" # Change label for demo
            
            detections.append({
                "box": box.xyxy[0].tolist(),
                "confidence": conf,
                "class": label
            })

    return {
        "detections": detections, 
        "alert_triggered": alert_triggered, 
        "count": len(detections)
    }
    

def draw_bounding_boxes(image_np, detections):
    img_copy = image_np.copy()
    for det in detections:
        box = det['box']
        label = det['class']
        conf = det['confidence']
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f"{label} {conf:.2%}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)