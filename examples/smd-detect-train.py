from yolo26mlx import YOLO
import mlx.core as mx

mx.disable_compile()

# Load pretrained MLX weights
model = YOLO("models/yolo26n.npz")
results = model.train(
     data="/Volumes/media-backup/AI-data/pcb_detection/smd-component-detection.v13i.yolo26/data-yolo26-mlx.yaml",       # dataset name or path to data YAML
     epochs=100,
     batch=4,
     imgsz=640,
     project="runs/smd-component-detection-train",
     name="my_smd-component-detection",
)