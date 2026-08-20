# ♻️ Smart Waste Sorting

### AI-powered waste detection, segmentation & real-time sorting analytics — built with Ultralytics YOLO

> *Because sorting trash shouldn't be a guessing game.* 🗑️➡️🤖➡️♻️

---

## 🌍 The Problem

Every day, tons of recyclable material end up in the wrong bin simply because sorting it by hand is slow, inconsistent, and honestly... nobody's favorite job. **Smart Waste Sorting** puts computer vision to work instead — spotting, segmenting, tracking, and counting waste items in real footage from an actual material recovery facility.

No stock photos. No toy examples. Real conveyor belt, real trash, real numbers.

---

## ✨ What This Project Does

| Capability | Description |
|---|---|
| 🔍 **Detection & Segmentation** | Pixel-level instance segmentation — not just boxes, actual object outlines |
| 🎯 **Custom-Trained Model** | Fine-tuned YOLO on a real solid-waste dataset (5 waste categories) |
| 🎥 **Video Analytics** | Object tracking + region-based counting on live conveyor-belt footage |
| 📊 **Model Evaluation** | mAP, precision/recall per class, with honest interpretation of where the model struggles |
| 🚀 **Zero-Setup Reproducibility** | Anyone can open the notebook, hit *Run All*, and watch it work — video and weights download automatically |

---

## 🧠 Tech Stack

- **[Ultralytics YOLO](https://ultralytics.com/)** — detection, segmentation, tracking & counting
- **OpenCV** — video I/O and frame-by-frame processing
- **Roboflow** — dataset hosting & versioning
- **yt-dlp** — automated video sourcing
- **Google Colab** — training & experimentation environment


---

## 🗂️ The Dataset

Trained on the **[Solid Waste dataset](https://universe.roboflow.com/xaviervape-old/solid-waste-ajntx)** via Roboflow Universe, covering 5 real-world waste categories:

`General trash` · `Glass` · `Paper pack` · `Plastic bag` · `Plastic`

---

## 🏋️ Training Summary

| Setting | Value |
|---|---|
| Base model | `yolov8n-seg.pt` |
| Epochs | 60 (early-stopped at 54) |
| Image size | 640×640 |
| Batch size | 16 |
| Frozen layers | 10 |

---

## 📈 Evaluation — The Honest Version

We didn't just run `model.val()` and call it a day — here's what the numbers actually mean:

| Class | mAP50 | What's happening |
|---|---|---|
| 🥃 Glass | ~0.67 | Best performer — plenty of training examples |
| 🧴 Plastic | ~0.54 | Solid, despite fewer samples |
| 📦 Paper pack | ~0.32 | Room to grow |
| 🗑️ General trash | ~0.02 | Struggles hard — only 3 training instances |
| 🛍️ Plastic bag | ~0.02 | Same story — 5 training instances, mostly missed (false negatives) |

**Takeaway:** the model is only as good as the data behind each class. Glass and Plastic had enough examples to learn real patterns; General trash and Plastic bag didn't — that's a data problem, not an architecture problem, and the fix is more (and more varied) labeled examples for those categories.

Confidence threshold: `0.25` · IoU threshold: `0.7` (Ultralytics defaults, chosen to balance catching true detections without flooding false positives).

---

## 🎬 Real-World Video Analytics

This is where it gets fun. Instead of running inference on a single image and calling it done, the pipeline:

1. 📥 Pulls real conveyor-belt footage from a recycling facility (auto-downloaded, no manual steps)
2. 🎯 Tracks each waste item frame-by-frame with a persistent ID
3. 📏 Counts items as they cross a defined "sorting line"
4. 💾 Saves the fully annotated video as proof it actually ran


---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install ultralytics opencv-python yt-dlp
```

### 2. Open the notebook
Open `smart_waste_sorting.ipynb` in Google Colab or Jupyter.

### 3. Run it
Hit **Run All**. That's genuinely it — the video and trained model weights download automatically from this repo, no manual file uploads required.

### 4. Check the output
Look for `output_sorted.mp4` and the per-class count printed at the end of the video analytics section.

---

## 🎓 Training Program Attribution

This project was built as the capstone for:

**Computer Vision for Developers **
*SDAIA Academy* — delivered via DAICO

🔗 [github.com/SDAIAAcademy](https://github.com/SDAIAAcademy)

---

## 👥 Team

| Name | Contribution |
|---|---|
| _Sarah Al Saed_ | _Real-World Solution & Video Analytics_ |
| _Noura Alfaadhel_ | _Core Vision Task & Interface_ |
| _Ghala Alharbi_ | _Custom Data & Training_ |
| _Sarah Alkhudhiri_ | _Model Evaluation_ |
| _Raneem Alsheddi_ | _Deployment & Export_ | 

---

## 🔮 What's Next

- 📦 Export to ONNX/TFLite for edge deployment on an actual sorting line
- 🖥️ Wrap it in a small Streamlit dashboard for live monitoring
- 🗂️ Grow the dataset for the underperforming classes (General trash, Plastic bag)

---
