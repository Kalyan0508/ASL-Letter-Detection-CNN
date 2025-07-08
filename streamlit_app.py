import streamlit as st
import torch
import sys
import os
import numpy as np
import cv2
from PIL import Image
import pathlib

# 🔧 Fix for Windows loading Unix-trained model (PosixPath issue)
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Add YOLOv5 to path (adjust if needed)
YOLOV5_PATH = os.path.join(os.getcwd(), 'yolov5')
sys.path.insert(0, YOLOV5_PATH)

# ✅ Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# 🔌 Load model
device = select_device('cpu')  # or 'cuda:0' for GPU
model = DetectMultiBackend('weights/best.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt

# 🖼️ Streamlit UI
st.title("🤟 ASL Letter Detection using YOLOv5")
st.write("Upload an image of a hand showing an ASL gesture (A–Z).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image using PIL, convert to RGB and NumPy
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Preprocess using YOLOv5 letterbox
    img_preprocessed = letterbox(img, new_shape=640, stride=stride, auto=True)[0]
    img_preprocessed = img_preprocessed.transpose((2, 0, 1))  # HWC to CHW
    img_preprocessed = np.ascontiguousarray(img_preprocessed)

    # Convert to tensor
    img_tensor = torch.from_numpy(img_preprocessed).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 🔍 Inference
    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # 🖊️ Draw results
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()

            for *xyxy, conf, cls in det:
                label = f"{names[int(cls)]} {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = cv2.putText(img, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show result
    st.image(img, caption="Detected ASL Letter", use_column_width=True)

    # Raw output (optional)
    detections = []
    for *xyxy, conf, cls in det:
        detections.append({
            "letter": names[int(cls)],
            "confidence": float(conf),
            "bbox": [int(x.item()) for x in xyxy]
        })

    st.subheader("Prediction Results")
    st.json(detections)
