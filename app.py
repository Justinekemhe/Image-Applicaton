from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel
from transformers import AutoModelForObjectDetection, DetrImageProcessor
from PIL import Image
import io
import torch
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; change to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods; adjust as needed
    allow_headers=["*"],  # Allows all headers; adjust as needed
)

# Replace with your actual Hugging Face token
token = "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"
model_name = "smutuvi/flower_count_model"

# Load model and processor
model = AutoModelForObjectDetection.from_pretrained(model_name, token=token)
processor = DetrImageProcessor.from_pretrained(model_name, token=token)

class ImageResponse(BaseModel):
    filename: str
    count: int
    error: str = None

@app.post("/batch_predict/")
async def batch_predict(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            logging.info(f"Processing file: {file.filename}")
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Example logic to count flowers based on bounding boxes
            bounding_boxes = outputs.pred_boxes
            count = len(bounding_boxes)  # Count of detected objects
            
            results.append({"filename": file.filename, "count": count})
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "error": str(e)})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
