from yolo26mlx import YOLO
# from safetensors.torch import load_file

# 读取所有权重，得到字典 key: tensor
# weights = load_file(
#     "runs/smd-component-detection-train/exp_A_1/best.safetensors", device="mps"
# )  # 可以 device="cpu"
# print(list(weights.keys())[:10])  # 打印前10个权重名，看结构

# model = YOLO("models/yolo26n.npz", task="pose")
model = YOLO("runs/smd-component-detection-train/exp_A_1/best.safetensors")
# model._load_safetensors(weights)  # 将权重加载到模型中
model.names = {
    0: 'CAPACITOR',
    1: 'DIODE',
    2: 'EC',
    3: 'IC',
    4: 'LED',
    5: 'RESISTOR',
    6: 'SCAPACITOR',
    7: 'ZENER',
}

results = model.predict(
    "/Volumes/media-backup/AI-data/pcb_detection/smd-component-detection.v13i.yolo26/images/test",
    imgsz=640,
    conf=0.25,
    save=True
)
print(results[0])  # detection + keypoint summary
results[0].save()  # saves annotated image with skeleton overlays to results/
