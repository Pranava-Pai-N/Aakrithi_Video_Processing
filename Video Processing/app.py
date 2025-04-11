from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
from PIL import Image
from ModelCode import classify_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


@app.get("/")
async def root():
    return {"message": "Welcome to Video Classification API!"}


@app.post("/Video_Processing")
async def process_video(file: UploadFile = File(...)):
    try:
        file_path = f"./{file.filename}"
        if not file.filename.endswith(('.mp4', '.avi', '.mov')):
            return {"error": "File is not a supported video format"}

        with open(file_path, "wb") as video_file:
            video_file.write(await file.read())

        result = classify_video(file_path)

        return {
            "Type": result["Type"],
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
