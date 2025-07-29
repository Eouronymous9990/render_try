from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from .yolo_model import model

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "YOLOv8 Object Detection API running on Render!"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    detections = results[0].boxes.data.tolist()

    return JSONResponse({"detections": detections})
