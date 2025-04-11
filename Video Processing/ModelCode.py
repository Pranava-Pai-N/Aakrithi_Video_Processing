import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
from collections import Counter

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % (fps * frame_rate)) == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
        count += 1

    cap.release()
    return frames

def classify_frame_with_clip(image):
    texts = ["Ayurveda", "Non-Ayurveda"]
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return texts[pred]

def classify_video(video_path):
    frames = extract_frames(video_path, frame_rate=5)  # Fewer frames for speed

    clip_preds = []

    for frame in frames:
        clip_result = classify_frame_with_clip(frame)
        clip_preds.append(clip_result)

    final_pred = Counter(clip_preds).most_common(1)[0][0]
    return {"Type": final_pred}
