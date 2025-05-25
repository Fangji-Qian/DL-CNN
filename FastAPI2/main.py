import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision import transforms, models
from ultralytics import YOLO

from gemini_client import build_prompt, call_gemini

app = FastAPI()

# Configuring static and template html file directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set model path
DETECT_WEIGHTS = "yolo_best.pt"
CLASSIFY_WEIGHTS = "ResNet34_best.pt"
CSV_PATH = "dataset/train.csv"

# Creating an upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLO model
detect_model = YOLO(DETECT_WEIGHTS)
detect_model.conf = 0.2

# Load the label and ResNet-34 Model
df = pd.read_csv(CSV_PATH)
species_col = next((col for col in ["snake_sub_family", "snake_sub", "binomial"] if col in df.columns), None)
if species_col is None:
    raise RuntimeError("Missing snake_sub_family / snake_sub / binomial column in CSV")

class_names = sorted(df[species_col].unique().tolist())
num_classes = len(class_names)

# Constructing a poisonous mapping table
poisonous_map = dict(zip(df[species_col], df["poisonous"]))

def build_cls_model(nc):
    m = models.resnet34(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, nc)
    return m

cls_model = build_cls_model(num_classes)
cls_model.load_state_dict(torch.load(CLASSIFY_WEIGHTS, map_location="cpu"))
cls_model.eval()

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Default first aid advice (when LLM interface is not available)）
DEFAULT_ADVICE = {
    True: [
        "Remain calm and limit movement of the injured limb",
        "Apply a pressure bandage to the entire limb from the bite site",
        "Elevate the injured limb to heart level",
        "Seek medical attention immediately and bring a photo or specimen of the snake"
    ],
    False: [
        "Wash the wound with clean water and soap",
        "Apply light bandage to prevent infection",
        "Observe for 24-48 hours, and seek medical attention if redness, swelling or fever occurs"
    ]
}

# Home page(Upload image)
@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Image Prediction and HTML Return
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Unable to decode image")

    # YOLO Detection
    det = detect_model(img)[0]
    if len(det.boxes) == 0:
        raise HTTPException(404, "No snake detected")

    best_box = det.boxes[det.boxes.conf.argmax()]
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    crop = img[y1:y2, x1:x2]

    #ResNet34 Classification
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = cls_model(tensor)
        probs = torch.softmax(logits, dim=1)
        idx = int(probs.argmax())
        species = class_names[idx]

    # Determining poisonousness
    venomous = bool(poisonous_map.get(species, 0))

    # Find additional information on this species
    row = df[df[species_col] == species].iloc[0]

    extra_info = {
        "country": row.get("country", "Unknown"),
        "continent": row.get("continent", "Unknown"),
        "genus": row.get("genus", "Unknown"),
        "family": row.get("family", "Unknown")
    }

    # LLM inference
    prompt = build_prompt(species, venomous)
    try:
        info = call_gemini(prompt)
        if info:
            advice = info.get("first_aid", DEFAULT_ADVICE[venomous])
        else:
            advice = DEFAULT_ADVICE[venomous]
    except Exception:
        advice = DEFAULT_ADVICE[venomous]

    # Save image for display
    filename = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(save_path, "wb") as f:
        f.write(data)

    uploaded_image_url = f"/static/uploads/{filename}"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "uploaded_image_url": uploaded_image_url,
        "species": species,
        "venomous": venomous,
        "first_aid": advice,
        **extra_info
    })