# 🤟 ASL Letter Detection using YOLOv5 + Streamlit

<div align="center">
  <img src="https://img.shields.io/github/languages/top/Kalyan0508/ASL-Letter-Detection-CNN" />
  <img src="https://img.shields.io/github/last-commit/Kalyan0508/ASL-Letter-Detection-CNN" />
  <img src="https://img.shields.io/badge/Model-YOLOv5m-blue" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-orange" />
</div>

---

## 📌 Overview

This project detects static **American Sign Language (ASL)** letters (A–Z except J and Z) using a custom-trained **YOLOv5m model**. It features a clean and interactive web interface built with **Streamlit**, allowing users to upload an image and get instant predictions with bounding boxes and labels.

> 💡 Built to support communication for the hearing and speech-impaired community through real-time gesture recognition.

---

## 🚀 Features

- 🧠 Deep learning with YOLOv5
- 📷 ASL hand gesture detection (24 static letters)
- 🌐 Streamlit-based web app
- 🛠️ CPU-compatible (no GPU required)
- 🎯 High accuracy with custom-trained weights

---

## 🧠 Model Details

- **Model**: YOLOv5m (custom trained)
- **Classes**: A–Z (excluding J and Z due to motion)
- **Framework**: PyTorch
- **Accuracy**: ~98% validation accuracy
- **Input size**: 640×640
- **Output**: Bounding box + label for each detected letter

---

## 📁 Project Structure

- ASL-Letter-Detection-CNN/
- ├── streamlit_app.py          # 🚀 Main Streamlit application
- ├── requirements.txt          # 📦 Python dependencies
- ├── packages.txt              # 🛠️ System-level dependencies for HF Spaces
- ├── yolov5/                   # 🧠 YOLOv5 model code (cloned/copied)
- ├── weights/                  # 🎯 Contains best.pt (trained model)
- ├── images/                   # 🖼️ Detection results (optional)
- ├── sample_images/            # 📂 Sample test images
- ├── notebook.ipynb            # 📓 Jupyter notebook for training/testing
- └── README.md                 # 📘 Project documentation



---

## 🛠️ Installation & Usage

```bash

## 🔧 1. Clone this Repository


git clone https://github.com/Kalyan0508/ASL-Letter-Detection-CNN.git
cd ASL-Letter-Detection-CNN


## 🧪 2. Install Dependencies
pip install -r requirements.txt

✅ Ensure your Python version is 3.8–3.11
🧠 Compatible with Torch 2.x and YOLOv5 v7.0+

## 🚀 3. Run the Web App
streamlit run streamlit_app.py

Upload an ASL hand gesture image to get predictions in real time!

---

## 🖼️ Sample Prediction

| Uploaded Image | Detected Output |
|----------------|------------------|
| ![Input](sample_images/sample1.jpg) | ![Detected](sample_images/sample1_result.jpg) |

> *(Replace with your actual image paths once added)*

---
## 🚀 Live Demo

👉 Try the project live on Hugging Face Spaces:
    ASL_Letter_Detection/
    ├── app.py                        # 🚀 Streamlit main app file
    ├── requirements.txt             # 📦 Python dependencies
    ├── yolov5/                      # 🧠 YOLOv5 core code (cloned or copied)
    │   ├── models/
    │   ├── utils/
    │   ├── ...                      # YOLOv5 scripts and modules
    ├── weights/
    │   └── best.pt                  # 🎯 Trained YOLOv5 model
    ├── README.md                    # 📄 Project documentation

   
🔗 ASL Letter Detection – Live Demo

   Link: https://huggingface.co/spaces/Kalyan0508/ASL_Letter_Detection

This interactive web app allows users to upload hand gesture images and detect the corresponding ASL (American Sign Language) letter using a custom-trained YOLOv5 model — all in real time.


---

## 📚 References

- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [Streamlit Docs](https://docs.streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [ASL Dataset - Roboflow Universe](https://universe.roboflow.com/)

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use or extend it — just give proper attribution 💙

---

> Built with ❤️ by **[Kalyan G](https://github.com/Kalyan0508)** — Making AI accessible and inclusive.



=======
# ASL-Letter-Detection-CNN
A CNN-based deep learning model to detect American Sign Language (ASL) letters from hand gesture images.
