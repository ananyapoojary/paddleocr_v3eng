from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os

# Initialize FastAPI app
app = FastAPI()

# Serve favicon to avoid 404 for browser favicon request
@app.get("/")
def read_root():
    return {"🎉 Welcome to the OCR API!"}

# OCR endpoint for POST request
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    from ocr_utils import run_ocr_on_image  # Import OCR function
    result = run_ocr_on_image(image_bytes)
    return JSONResponse(content=result)
