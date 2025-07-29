# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO("yolov8n.pt")  # أو المسار لموديلك

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    return {"detections": boxes}
