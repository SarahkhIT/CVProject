# Technical Documentation
## Solid Waste Instance Segmentation & Analysis Pipeline

---

# 1. System Overview

This project implements an end-to-end computer vision and machine learning pipeline designed for automated solid waste detection, classification, and instance segmentation. 

The system utilizes an object detection and segmentation framework built on top of the Ultralytics YOLO ecosystem. It ingests visual data, processes dataset configurations, trains instance segmentation models across isolated execution runs, tracks iteration performance metrics, and extracts quantitative bounding box coordinates and segmentation masks for waste identification tasks.

---

# 2. Architecture

The architecture follows a modular machine learning pipeline structure executed in a GPU-accelerated runtime context.

```
          Visual Data Input
                  │
                  ▼
      Dataset Configuration (YAML)
                  │
                  ▼
        YOLO Model Initialization
                  │
                  ▼
   Segmentation Training & Iteration
                  │
                  ▼
      Metrics Logging (results.csv)
                  │
                  ▼
   Inference & Post-Processing (Boxes/Masks)
```

---

# 3. Pipeline Modules & Components

## Component 1 — Dataset Module (`Solid-Waste-1`)

**Purpose**

Defines and parses dataset parameters, image paths, label paths, and class definitions required for waste segmentation.

**Input**

- Image assets and target mask labels

**Output**

- `Solid-Waste-1/data.yaml` configuration mapping

---

## Component 2 — Model Engine (`YOLO`)

**Purpose**

Handles initial weights loading, deep learning architecture setup, loss function computation, and execution of forward/backward passes during training.

**Frameworks**

- PyTorch (`torch` v2.11.0+cu128)
- Ultralytics YOLO primitives
- Hardware acceleration via NVIDIA CUDA

---

## Component 3 — Training & Artifact Tracker (`runs/segment/`)

**Purpose**

Isolates individual experimental runs, saves hyperparameter states, and logs epoch-by-epoch training and validation loss values.

**Outputs**

- Run configuration: `args.yaml`
- Execution logs: `results.csv`
- Saved model checkpoints

---

## Component 4 — Inference & Prediction Parser

**Purpose**

Runs trained models over target test assets, generating predictions wrapped inside `Results` objects (`r`) to extract bounding boxes (`box`) and spatial segmentation masks.

---

# 4. State Management & Run Isolation

Experimental runs are partitioned into dedicated directories under the `runs/` tracking tree to preserve model history and configuration integrity.

Key tracking directories:

```
runs/segment/waste_seg_runs/solid_waste_seg_v1/
runs/segment/waste_seg_runs/solid_waste_seg_v1-2/
```

Inside each run directory, the pipeline maintains:

- `args.yaml`: Stores exact hyperparameters, image size, learning rates, and target batch sizes.
- `results.csv`: Tracks continuous performance metrics including precision, recall, mAP50, and mAP50-95 across training epochs.

---

# 5. Technical Decisions

- **YOLO Segmentation Framework:** Selected for state-of-the-art balance between real-time inference speed and accurate pixel-level spatial masks required for complex waste items.
- **PyTorch & CUDA Backend:** Utilizes `torch==2.11.0+cu128` to ensure compute efficiency and tensor acceleration on CUDA-enabled GPU hardware.
- **YAML-Based Configs:** Decouples dataset paths and model execution settings from code logic, allowing continuous iteration on data splits without source code modification.
- **Modular Output Extraction:** Decouples raw predictions into dedicated `Results` and `Boxes` objects (`r`, `box`) to streamline downstream analytics and spatial overlay rendering.

---

# 6. Technologies Used

| Component | Technology | Version / Spec |
|------------|------------|----------------|
| Operating Language | Python | 3.12.13 |
| Deep Learning Framework | PyTorch | 2.11.0+cu128 |
| Vision Framework | Ultralytics YOLO | Native Class API |
| Training Accelerator | Accelerate | 1.14.0 |
| Datasets Library | Datasets | 4.0.0 |
| Model Fine-Tuning Support | PEFT | 0.20.0 |
| Supporting AI Frameworks | LangChain / Transformers | 1.3.15 / 5.15.0 |
| Execution Hardware | NVIDIA GPU | Tesla T4 |

---

# 7. Project Structure

```
Solid-Waste-Segmentation/

│── Solid-Waste-1/
│   └── data.yaml
│── runs/
│   └── segment/
│       └── waste_seg_runs/
│           ├── solid_waste_seg_v1/
│           │   ├── args.yaml
│           │   └── results.csv
│           └── solid_waste_seg_v1-2/
│               ├── args.yaml
│               └── results.csv
│── project_notebook.ipynb
│── requirements.txt
│── README.md
```

---

# 8. Execution Flow

1. **Environment Initialization:** Load Python dependencies and establish CUDA connection with the Tesla T4 GPU.
2. **Data Pipeline Registration:** Read `Solid-Waste-1/data.yaml` to register class boundaries and data paths.
3. **Model Instantiation:** Load the target YOLO segmentation model class in memory.
4. **Training Execution:** Run multi-epoch training pass, creating output logs in `runs/segment/waste_seg_runs/`.
5. **Metric Logging:** Append training loss, validation loss, and precision metrics into `results.csv`.
6. **Inference & Mask Extraction:** Output predicted bounding boxes (`box`) and visual segmentation contours (`r`).

---

# 9. Production Readiness

- **Modular Directory Organization:** Isolated training runs avoid configuration overwrites and provide reproducible metrics.
- **GPU Acceleration:** Optimized for low-latency instance segmentation pipelines using CUDA bindings.
- **Auditability:** Complete snapshot of hyperparameters preserved per experiment via `args.yaml`.
- **Hardware Agnostic Data Loading:** Standardized dataset specifications in YAML format.

---

# 10. Future Improvements

- Automated deployment of the segmentation model behind a REST API (FastAPI/Flask).
- Edge device export (TensorRT or ONNX format) for real-time robotic or conveyor belt sorting.
- Real-time video stream processing and counting dashboard.
- Integration with cloud storage (S3 / GCP Buckets) for continuous dataset logging and auto-retraining.

