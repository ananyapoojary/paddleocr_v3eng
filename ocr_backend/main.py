from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os

# Initialize the app once
app = FastAPI()

# Serve favicon to stop 404 in browser
@app.get("/")
def read_root():
    return {"message": "🎉 Welcome to the OCR API! Use /docs to try it out."}


# OCR endpoint
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    from ocr_utils import run_ocr_on_image  # Import here to avoid circular import issues if needed
    result = run_ocr_on_image(image_bytes)
    return result
