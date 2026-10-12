from yolo26mlx import YOLO
import mlx.core as mx
mx.disable_compile()

model = YOLO("models/yolo26n.npz")
results = model.train(
    data="./configs/smd-detect-yolo26-no-mosaic.yaml",
    epochs=30,
    batch=4,
    imgsz=640,
    project="runs/smd-component-detection-train",
    name="exp_A_1",
    patience=5,
)
