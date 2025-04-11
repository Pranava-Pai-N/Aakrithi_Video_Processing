import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms
from pytesseract import image_to_string
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter
from pytesseract import pytesseract

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)
blip_model = blip_model.to(device)

def extract_frames(video_path, frame_rate=1):
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

def get_caption_with_blip(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def extract_text_with_ocr(image):
    return image_to_string(image)

def classify_video(video_path):
    frames = extract_frames(video_path, frame_rate=2)

    clip_preds = []
    blip_preds = []
    ocr_preds = []

    for frame in frames:
        clip_result = classify_frame_with_clip(frame)
        clip_preds.append(clip_result)

        caption = get_caption_with_blip(frame)
        blip_input = clip_processor(text=["Ayurveda", "Non-Ayurveda"], images=frame, return_tensors="pt", padding=True).to(device)
        blip_output = clip_model(**blip_input)
        blip_probs = blip_output.logits_per_image.softmax(dim=1)
        blip_pred = torch.argmax(blip_probs, dim=1).item()
        blip_preds.append(["Ayurveda", "Non-Ayurveda"][blip_pred])

        text = extract_text_with_ocr(frame)
        if any(keyword in text.lower() for keyword in ["ayurveda", "herbal", "vedic", "naturopathy"]):
            ocr_preds.append("Ayurveda")
        else:
            ocr_preds.append("Non-Ayurveda")

    all_preds = clip_preds + blip_preds + ocr_preds
    final_pred = Counter(all_preds).most_common(1)[0][0]
    return {"Type": final_pred}