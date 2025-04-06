# inference/api.py
from fastapi import FastAPI, UploadFile, File
import shutil, os
from main import run

# run using the below command
# uvicorn api:app --host 0.0.0.0 --port 8000
app = FastAPI()

@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    input_dir = "D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/output"
    file_path = os.path.join(input_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = run(file_path)
    return {
        "status": "success",
        "output_pdf": file.filename,
        "extracted_text": extracted_text
    }
