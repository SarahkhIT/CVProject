# Smart Waste Sorting

### AI-powered waste detection, segmentation & real-time sorting analytics — built with Ultralytics YOLO

---

## The Problem

Every day, tons of recyclable material end up in the wrong bin because sorting it by hand is slow, inconsistent, and expensive to scale. **Smart Waste Sorting** applies computer vision instead — detecting, segmenting, tracking, and counting waste items directly from video, as a step toward automated sorting on a real conveyor line.

---

## What This Project Does

| Capability | Description |
|---|---|
| **Detection & Segmentation** | Pixel-level instance segmentation — actual object outlines, not just boxes |
| **Custom-Trained Model** | Fine-tuned YOLO on a real solid-waste dataset (9 waste categories, from a snapshot of a larger 16-class Roboflow project) |
| **Video Analytics** | Object tracking + region-based counting over real video footage |
| **Model Evaluation** | mAP, precision/recall per class, with an honest interpretation of where the model struggles |
| **Deployment-Ready Export** | Model exported to ONNX for cross-platform, edge-friendly inference |
| **Reproducible End-to-End** | Open the notebook, run it top to bottom — sample images, video, and model weights are fetched automatically |

---

## Tech Stack

- **[Ultralytics YOLO](https://ultralytics.com/)** — detection, segmentation, tracking & counting
- **OpenCV** — video I/O and frame-by-frame processing
- **Roboflow** — dataset hosting & versioning
- **yt-dlp** — automated video sourcing
- **ONNX** — model export for deployment
- **Google Colab** — training & experimentation environment

---

## The Dataset

Trained on the **[Solid Waste dataset](https://universe.roboflow.com/xaviervape-old/solid-waste-ajntx)** via Roboflow Universe. The notebook pulls `project.version(1)`, a fixed snapshot with **9 waste categories, 113 training images and 10 validation images**. Class representation within that snapshot is uneven (e.g. Glass has far more labeled instances than General Trash or Plastic Bag), which directly shapes the evaluation results below.

**Note:** the live Roboflow project has since grown to **16 classes and 979 images**. Because `version(1)` is a frozen snapshot, the model in this repo was trained and evaluated only on the original 9-class version — not the current, larger dataset. Training on the full 16-class version is a natural next step (see "What's Next" below) but was not done for this submission.

---

## Training Summary

| Setting | Value |
|---|---|
| Base model | `yolov8n-seg.pt` |
| Epochs | 60 (early-stopped at 54, best checkpoint at 39) |
| Image size | 640×640 |
| Batch size | 16 |
| Frozen layers | 10 |

**Overall result:** mAP50 = 0.319 / mAP50-95 = 0.228 (box), mAP50 = 0.309 / mAP50-95 = 0.201 (mask).

---

## Evaluation — The Honest Version

We didn't just run `model.val()` and call it done — here's what the numbers actually mean:

| Class | Validation instances | mAP50 | What's happening |
|---|---|---|---|
| Glass | 45 | ~0.69 | Best performer — plenty of training examples |
| Plastic | 4 | ~0.55 | Strong despite few samples |
| General trash | 2–5 | ~0.02 | Fails hard — too few examples to learn from |
| Plastic bag | 2–5 | ~0.02 | Same story — mostly missed (false negatives) |

*(Table shows the classes with the most representative results out of the 9 classes in the trained snapshot.)*

**Takeaway:** this isn't classic overfitting (train loss down, val loss up) — it's underfitting driven by class imbalance. With only 113 training images spread across 9 classes, well-represented classes like Glass learned real patterns, while minority classes simply didn't have enough examples. More (and more varied) labeled data for the weak classes is the fix, not a different architecture.

Confidence threshold: `0.25` · IoU threshold: `0.70`, chosen using Ultralytics' defaults and validated by comparing precision/recall across multiple thresholds (0.15–0.50) in the evaluation section.

---

## Real-World Video Analytics

Rather than stopping at single-image inference, the pipeline also:

1. Downloads real video footage automatically (via `yt-dlp`, no manual steps)
2. Tracks each waste item frame-by-frame using Ultralytics' `ObjectCounter`
3. Counts items as they cross a defined region/line
4. Saves the fully annotated output video as evidence it actually ran

---

## Deployment & Export

The trained model is exported to **ONNX** for cross-platform, hardware-independent inference (CPU, GPU, or edge devices), with graph-level optimizations that improve inference speed — important for a waste-sorting system that may eventually need to run in real time on lower-power hardware. The exported model is verified with a test inference pass before being considered deployment-ready.

---

## Getting Started

### 1. Install dependencies
```bash
pip install ultralytics opencv-python yt-dlp roboflow
```

### 2. Open the notebook
Open `smart_waste_sorting.ipynb` in Google Colab or Jupyter.

### 3. Run it
Run all cells top to bottom. Sample images, video, and trained model weights are fetched automatically — no manual file uploads required. A Roboflow API key (`ROBOFLOW_KEY`) is needed only if you want to re-run training from scratch.

### 4. Check the output
Look for output_sorted.mp4 (annotated video) and the per-class detection counts printed at the end of the video analytics section. Trained weights (best.pt) are included directly in this repository — no separate download needed.

---

## Training Program Attribution

This project was built as the capstone for:

**Computer Vision for Developers**
*SDAIA Academy* — delivered via DAICO

*Trainer:* Mohammad Albeladi
*Session Dates:* 16th of August, 2026 – 20th of August, 2026

🔗 [github.com/SDAIAAcademy](https://github.com/SDAIAAcademy)

---

## Team

| Name | Contribution |
|---|---|
| *Sarah Al Saed* | *Real-World Solution & Video Analytics* |
| *Noura Alfaadhel* | *Core Vision Task & Interface* |
| *Ghala Alharbi* | *Custom Data & Training* |
| *Sarah Alkhudhiri* | *Model Evaluation* |
| *Raneem Alsheddi* | *Deployment & Export* |

---

## What's Next

- Retrain on the current, larger version of the dataset (16 classes, 979 images) instead of the original 9-class snapshot (`version(1)`)
- Grow labeled examples for underperforming classes (General trash, Plastic bag)
- Wrap the pipeline in a small Streamlit dashboard for live monitoring
- Benchmark the ONNX export on actual edge hardware for a real sorting line
