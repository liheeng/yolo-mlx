from yolo26mlx import YOLO
import mlx.core as mx
mx.disable_compile()

model = YOLO("models/yolo26n.npz")
results = model.train(
    data="/Volumes/media-backup/AI-data/pcb_detection/smd-component-detection.v13i.yolo26/data-yolo26-mlx.yaml",
    epochs=100,
    batch=4,
    imgsz=640,
    project="runs/smd-component-detection-train",
    name="exp_B_1",           # exp_B_1 / exp_B_2 / exp_B_3
    seed=42,
    # mosaic / mixup / lr0/lrf 全部保持基线原始值！不要改成方案A
)
