from ultralytics import YOLO


def main():
    # YOLOv8x-seg: largest variant of the YOLOv8 family for instance segmentation
    model = YOLO("yolov8x-seg.pt")

    results = model.train(
        data="vs_seg_dataset.yaml",
        epochs=500,              # number of loops through the entire dataset
        imgsz=640,               # input image size (640x640)
        batch=16,                # samples processed before each weight update
        patience=100,            # early stopping: epochs without improvement
        save=True,
        save_period=25,
        device=0,
        workers=4,
        project="runs/vs_seg",
        name="yolov8x_seg_full",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",        # resolves to SGD (reference article default)
        lr0=0.01,                # initial learning rate
        lrf=0.01,                # final LR as a fraction of lr0
        momentum=0.9,            # SGD momentum
        weight_decay=0.0005,     # L2 regularization penalty
        seed=42,
        val=True,
        plots=True,
        verbose=True,
    )

    # Validate on test set using best checkpoint
    model_best = YOLO("runs/vs_seg/yolov8x_seg_full/weights/best.pt")
    metrics = model_best.val(
        data="vs_seg_dataset.yaml",
        split="test",
        verbose=True,
        plots=True,
    )

    print("\n=== Test Results ===")
    print(f"mAP50     : {metrics.seg.map50:.4f}")
    print(f"mAP50-95  : {metrics.seg.map:.4f}")
    print(f"Precision : {metrics.seg.mp:.4f}")
    print(f"Recall    : {metrics.seg.mr:.4f}")


if __name__ == "__main__":
    main()