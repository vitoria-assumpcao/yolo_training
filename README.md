# YOLO Training — Vestibular Schwannoma Segmentation

Support repository for training a YOLOv8x-seg model on preprocessed images from the **VS-SEG dataset**, applied to Vestibular Schwannoma (VS) detection and segmentation from MRI scans.

The images used here were preprocessed by a dedicated pipeline (see [automated-vs-organization](https://github.com/vitoria-assumpcao/automated-vs-organization)) that converts the original VS-SEG NIfTI volumes into a 2D YOLO-compatible format.

---

## Reference

Training hyperparameters follow the configuration described in:

> Kurt Pehlivanoğlu et al. (2025). *Towards advanced brain tumor segmentation: a novel hybrid architecture integrating UNet, FCN, and YOLO models on the newly introduced BTS-DS 2024 dataset*. Eur. Phys. J. Spec. Top., 234, 4421–4435. https://doi.org/10.1140/epjs/s11734-025-01698-6

---

## Repository Structure

```
yolo_training/
├── yolo_training/
│   ├── train_yolo.py          # Training and validation script
│   └── vs_seg_dataset.yaml    # Dataset configuration for YOLOv8
├── training_set_vs_tumor.zip  # Preprocessed training images (YOLO format)
├── pacientes_para_validação.zip  # Validation/test patient images
├── pyproject.toml
└── .gitignore
```

---

## Dataset

The dataset is derived from the publicly available **VS-SEG** dataset (Shapey et al., 2021), preprocessed into 2D slices in YOLO segmentation format (images + polygon masks).

Split configuration (`vs_seg_dataset.yaml`):
- `train/images` — training set
- `valid/images` — validation set
- `test/images` — test set

Single class: `tumor` (Vestibular Schwannoma).

---

## Model

**YOLOv8x-seg** — largest variant of the YOLOv8 family, adapted for instance segmentation tasks.

Pretrained weights: `yolov8x-seg.pt` (COCO, via Ultralytics).

---

## Hyperparameters

| Parameter      | Value  | Notes                              |
|----------------|--------|------------------------------------|
| `epochs`       | 500    | Early stop with patience=100       |
| `batch`        | 16     |                                    |
| `imgsz`        | 640    |                                    |
| `optimizer`    | auto   | Resolves to SGD                    |
| `lr0`          | 0.01   | Initial learning rate              |
| `lrf`          | 0.01   | Final LR as fraction of lr0        |
| `momentum`     | 0.9    | SGD momentum                       |
| `weight_decay` | 0.0005 |                                    |
| `patience`     | 100    | Early stopping                     |
| `seed`         | 42     |                                    |

---

## Requirements

```bash
pip install ultralytics
```

Or via Poetry:

```bash
poetry install
```

---

## Usage

```bash
python yolo_training/train_yolo.py
```

Results and checkpoints are saved to `runs/vs_seg/yolov8x_seg_full/`.