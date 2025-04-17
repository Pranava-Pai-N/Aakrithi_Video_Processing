import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms
from pytesseract import image_to_string
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


def extract_frames(video_path, frame_rate=1, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(fps * frame_rate) == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
        count += 1

    cap.release()
    return frames



def classify_frames_with_clip(images):
    texts = ["Ayurveda", "Non-Ayurveda"]
    inputs = clip_processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        preds = torch.argmax(probs, dim=1).tolist()

    return [texts[pred] for pred in preds]


def get_blip_predictions(images):
    predictions = []
    keywords = ["ayurveda", "herbal", "vedic", "naturopathy"]
    for image in images:
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True).lower()
        if any(kw in caption for kw in keywords):
            predictions.append("Ayurveda")
        else:
            predictions.append("Non-Ayurveda")
    return predictions


def get_ocr_predictions(images):
    predictions = []
    keywords = ["ayurveda", "herbal", "vedic", "naturopathy"]
    for image in images:
        text = image_to_string(image).lower()
        if any(kw in text for kw in keywords):
            predictions.append("Ayurveda")
        else:
            predictions.append("Non-Ayurveda")
    return predictions

def classify_video(video_path):
    frames = extract_frames(video_path, frame_rate=1, max_frames=10)

    clip_preds = classify_frames_with_clip(frames)

    blip_preds = get_blip_predictions(frames)
    ocr_preds = get_ocr_predictions(frames)

    all_preds = clip_preds + blip_preds + ocr_preds
    final_pred = Counter(all_preds).most_common(1)[0][0]

    return {"Type": final_pred}
