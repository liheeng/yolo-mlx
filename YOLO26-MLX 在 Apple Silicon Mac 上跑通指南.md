# YOLO26\-MLX 在 Apple Silicon Mac 上跑通指南

> [!NOTE]
>  [包含例子的yolo-mlx 项目地址](https://github.com/liheeng/yolo-mlx)

**一句话路径：** 创建 Conda 环境安装 MLX → 准备 YOLO 格式数据集并写好 data\.yaml → 加载 yolo26n\.npz 启动训练 → 加载 best\.safetensors 并手动覆盖 model\.names 后推理。

本文基于 yolo26\-mlx（纯 MLX 实现的 YOLO26）实操整理，覆盖检测 / 姿态任务，全部命令与脚本在 M 系列芯片上验证过。

# 1\. 环境部署

## 1\.1 硬件与系统要求

|项目|要求|
|---|---|
|芯片|Apple Silicon M1 及以上（M1/M2/M3/M4 Pro/Max/Ultra），依赖统一内存|
|系统|macOS 13\.3 及以上|
|内存|建议 16 GB 以上；imgsz=640 \+ batch=4 约占 8\~12 GB 统一内存|
|Python|3\.10 \~ 3\.13（推荐 3\.13）|
|MLX|推荐 0\.18\.x 稳定版；0\.20\+ 在训练循环中存在内存泄漏风险|

## 1\.2 安装步骤

```bash
git clone <yolo26-mlx 仓库地址>
cd yolo26-mlx

conda create -n yolo26-mlx python=3.13 -y
conda activate yolo26-mlx

pip install mlx==0.18.2 numpy pyyaml pillow opencv-python safetensors
pip install -e .
```

## 1\.3 安装验证

```python
import mlx.core as mx
from yolo26mlx import YOLO

print("MLX version:", mx.__version__)
model = YOLO("models/yolo26n.npz")
print("Default classes:", len(model.names))
```

正常输出 MLX 版本号与默认 COCO 类别数（80）即安装成功。

# 2\. 数据集准备

## 2\.1 目录结构

```text
datasets/
└── smd-component-detection/
    ├── images/
    │   ├── train/
    │   └── valid/
    └── labels/
        ├── train/
        └── valid/
```

标注为同名 `.txt`，检测任务每行：`class_id cx cy w h`（归一化 0\~1）；姿态任务追加 `kx1 ky1 v1 kx2 ky2 v2 ...`。

## 2\.2 data\.yaml 配置

**关键坑：** yolo26mlx 的 `model.train()` **不接受** `mosaic / mixup / seed / lrf` 等关键字参数，必须写在 data\.yaml 里，否则报 `TypeError: unexpected keyword argument`。

```yaml
# 任务与类别
task: detect
nc: 8
names: ["capacitor", "resistor", "ic", "diode",
        "inductor", "led", "connector", "fuse"]

# 数据集路径（建议绝对路径）
path: /Volumes/media-backup/AI-data/pcb_detection/smd-component-detection.v13i.yolo26
train: images/train
val: images/valid

# 数据增强（写在 yaml 里才生效）
mosaic: 0.0          # PCB 小元件建议关闭
mixup: 0.0
copy_paste: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
fliplr: 0.5
```

姿态任务追加：

```yaml
task: pose
kpt_shape: [4, 3]   # 关键点数量与维度，必须与标注一致
```

# 3\. 模型训练

## 3\.1 基础训练脚本

```python
from yolo26mlx import YOLO
import mlx.core as mx

# 调试阶段可关闭编译缓存，正式训练注释掉
# mx.disable_compile()

# 加载预训练基础模型（.npz）
model = YOLO("models/yolo26n.npz")

results = model.train(
    data="./configs/smd-detect.yaml",
    epochs=30,
    batch=4,
    imgsz=640,
    project="runs/smd-component-detection-train",
    name="exp_A_1",
    patience=5,   # 连续 5 epoch mAP 不提升则早停
)
print("Done:", results)
```

运行：

```bash
python examples/train_smd.py
```

## 3\.2 train\(\) 支持的参数

|参数|说明|注意事项|
|---|---|---|
|`data`|数据集 yaml 路径|支持相对/绝对路径|
|`epochs`|最大训练轮数|小数据集 30\~100 足够|
|`batch`|批次大小|统一内存有限，从 2\~4 起步|
|`imgsz`|输入尺寸|推荐 640|
|`project / name`|输出目录|结果保存在 project/name/|
|`patience`|早停轮数|防止过拟合|
|`lr`|覆盖自动学习率|不填则按 nc 自动计算|
|`freeze_bn`|冻结 BN 统计量|短训练默认 True，小 batch 必开|

## 3\.3 训练输出

训练完成后在 `runs/<project>/<name>/` 目录下：

- `best.safetensors`：验证集 mAP 最优权重

- `last.safetensors`：最后一轮权重

- `args.yaml`：本次训练完整参数（含 names）

## 3\.4 LR 调度说明

当前 trainer 实现的是**线性衰减**（不是 cosine），核心代码在 `trainer.py`：

```python
lrf = 0.01
lf = max(1 - epoch / epochs, 0) * (1.0 - lrf) + lrf
self.optimizer.learning_rate = self._lr0 * lf
```

即 warmup 3 epoch 后，从 `lr0` 线性降到 `lr0 * 0.01`。如需改成余弦退火，替换为：

```python
import math
end_lr = self._lr0 * 0.01
lf = end_lr + 0.5 * (self._lr0 - end_lr) * (
    1 + math.cos(math.pi * epoch / epochs)
)
self.optimizer.learning_rate = lf
```

# 4\. 模型推理与预测

## 4\.1 最关键的坑：类别名必须手动覆盖

**症状：** 加载 best\.safetensors 后，预测标签全部是 person / car / bus 等 COCO 类别，而不是你的 PCB 元器件。

**原因：**` YOLO("xxx.safetensors")` 只加载网络权重张量，**不写入类别名称表**，model\.names 仍是默认 COCO 80 类。

**解决：** 加载后从训练目录的 args\.yaml 读取 names 并覆盖 model\.names。

## 4\.2 完整推理脚本

```python
from yolo26mlx import YOLO
import yaml

# 1. 加载训练好的权重
model = YOLO("runs/smd-component-detection-train/exp_A_1/best.safetensors")

# 2. 关键：从训练配置恢复类别名称
with open("runs/smd-component-detection-train/exp_A_1/args.yaml") as f:
    train_args = yaml.safe_load(f)
model.names = {i: name for i, name in enumerate(train_args["names"])}

print("Classes:", model.names)  # 应输出你的元器件列表

# 3. 执行预测
results = model.predict(
    source="/Volumes/.../images/test",
    imgsz=640,
    conf=0.25,
    save=True,
)

print(results[0])
results[0].save()
```

如果训练目录没有 args\.yaml，也可以手动写死：

```python
model.names = {
    0: "capacitor", 1: "resistor", 2: "ic", 3: "diode",
    4: "inductor", 5: "led", 6: "connector", 7: "fuse",
}
```

## 4\.3 验证类别是否正确

```python
print(model.names)
# 错误输出：{0: 'person', 1: 'bicycle', ...}  →  没覆盖 names
# 正确输出：{0: 'capacitor', 1: 'resistor', ...}  →  OK
```

# 5\. 常见问题排查

## 5\.1 FileNotFoundError: model config not found: yolo26

CLI 自动拼接模型名失败。解决：不要用 `--scale` 自动拼接，直接用 Python 脚本加载 `models/yolo26n.npz` 完整路径。

## 5\.2 Metal Invalid Resource / 内存瞬间冲到 100 GB

典型出现于姿态任务或 batch 过大。按顺序尝试：

1. 把 batch 降到 2\~4

2. 设置环境变量关闭 MLX 内存池：

```bash
export MLX_DISABLE_MEMORY_POOL=1
export MLX_ALLOCATOR=system
```

1. 确认模型头与任务匹配：检测数据集不能用 pose 模型训练，反之亦然

2. MLX 降级到 0\.18\.2 稳定版

## 5\.3 训练 mAP 持续单边下跌

现象：前 2 epoch mAP 最高，之后一路下跌，Recall 断崖。常见原因与对策：

|原因|对策|
|---|---|
|Mosaic 增强过强，小元件标注错位|data\.yaml 中设 `mosaic: 0.0, mixup: 0.0`|
|one2many 分支匹配不稳定|在 `utils/loss.py` 注释 one2many loss，只保留 one2one|
|小 batch BN 统计量漂移|确认 `freeze_bn=True`（短训练默认开启）|

## 5\.4 预测标签全是 person / car

即第 4\.1 节的问题，手动覆盖 `model.names` 即可。

## 5\.5 train\(\) 报 unexpected keyword argument

yolo26mlx 的 train\(\) 函数参数列表有限，`mosaic / mixup / seed / lrf` 等都要写到 data\.yaml，不能通过 Python 函数参数传入。

# 6\. 消融实验 A 完整操作

消融实验 A 的定义：**关闭 Mosaic \+ MixUp \+ CopyPaste 多图融合强增强，保持框架原生线性学习率调度**。实验目的：验证多图融合强增强是否是 mAP 震荡、验证指标不稳定的根源。

## 6\.1 变量定义（严格单变量）

|条件|基线组（对照）|实验 A 组|说明|
|---|---|---|---|
|Mosaic|1\.0（开启）|0\.0（关闭）|**唯一变量**|
|MixUp|0\.1（开启）|0\.0（关闭）|跟随 Mosaic 一起关闭，保证变量纯净|
|CopyPaste|0\.1（开启）|0\.0（关闭）|跟随关闭|
|学习率调度|原生线性衰减|原生线性衰减（不变）|不修改 trainer\.py 的 LR 代码|
|模型结构 / 数据集 / 超参|完全相同|完全相同|nc、names、路径、hsv、fliplr、epochs、batch、imgsz、patience 全部一致|

## 6\.2 固定条件（基线组与实验 A 组必须全部保持一致）

1. 模型：yolo26n，one2one \+ one2many 双分支不变

2. 超参：imgsz=640，batch=4，epochs=30，patience=5

3. 学习率：使用框架原生线性衰减 LR，不修改训练器 LR 代码

4. 环境：两组训练前统一执行 MLX 内存环境变量；统一选择开启或关闭 `mx.disable_compile()`，不能一组开一组关

5. 其余增强（hsv、水平翻转）参数不变

## 6\.3 Step 1：基线对照组配置 smd\-detect\-baseline\.yaml

```yaml
path: /Volumes/media-backup/AI-data/pcb_detection/smd-component-detection.v13i.yolo26
train: images/train
val: images/valid
test: images/test
nc: 8
names: ["CAPACITOR", "DIODE", "EC", "IC", "LED", "RESISTOR", "SCAPACITOR", "ZENER"]

mosaic: 1.0
mixup: 0.1
copy_paste: 0.1
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
fliplr: 0.5
```

## 6\.4 Step 2：实验 A 组配置 smd\-detect\-no\-mosaic\.yaml

**复制基线 yaml**，只把增强参数改成 0，其余（nc、names、路径、test 目录）保持一致。上面这份是你实际使用的 `smd-detect-no-mosaic.yaml`。

**注意：**yaml 里写 `lr0 / lrf / warmup_epochs / seed` 训练器**不会读取**（trainer\.py 对 warmup、lrf 是硬编码，seed 不生效）；真正生效的只有增强参数。因此实验 A 的学习率仍是框架原生线性衰减，seed 无法固定。

```yaml
path: /Volumes/media-backup/AI-data/pcb_detection/smd-component-detection.v13i.yolo26
train: images/train
val: images/valid
test: images/test
nc: 8
names: ["CAPACITOR", "DIODE", "EC", "IC", "LED", "RESISTOR", "SCAPACITOR", "ZENER"]
# ========== 方案A 超参 ==========
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
lr0: 0.001
lrf: 1.0
warmup_epochs: 3
seed: 42
```

## 6\.5 训练脚本（仅切换 data 路径，代码不变）

```python
from yolo26mlx import YOLO
import mlx.core as mx
mx.disable_compile()
model = YOLO("models/yolo26n.npz")
results = model.train(
    data="./configs/smd-detect-baseline.yaml",
    epochs=30, batch=4, imgsz=640,
    project="runs/smd-component-detection-train",
    name="baseline_1", patience=5,
)
```

```python
from yolo26mlx import YOLO
import mlx.core as mx
mx.disable_compile()
model = YOLO("models/yolo26n.npz")
results = model.train(
    data="./configs/smd-detect-no-mosaic.yaml",
    epochs=30, batch=4, imgsz=640,
    project="runs/smd-component-detection-train",
    name="exp_A_1", patience=5,
)
```

## 6\.6 Step 3：训练前置环境（两组训练前都执行）

```bash
export MLX_DISABLE_MEMORY_POOL=1
export MLX_ALLOCATOR=system
python examples/train_smd.py
```

## 6\.7 Step 4：结果对比指标与结论判定

|指标|统计口径|结论判断|
|---|---|---|
|最佳 mAP50 均值|每组 3 次重复取平均|A 更高 = 关闭 Mosaic 对该任务有益|
|最佳 mAP50 标准差|每组 3 次重复求标准差|A 标准差更小 = Mosaic 是波动来源|
|最佳 epoch 均值|每组取平均|A 更早收敛 = Mosaic 延缓收敛|
|后 10 epoch mAP 波动幅度|最大值减最小值|A 波动小 = Mosaic 造成验证集震荡|

**结论对应：** mAP 小幅下降但波动明显变小，说明 Mosaic 有增益但引入不稳定性；mAP 基本不变且波动显著变小，说明 Mosaic 纯噪声可永久关闭；mAP 反而更高且波动更小，说明 Mosaic 破坏了 PCB 小元件标注一致性；两组差异很小，说明波动另有来源（one2many 分支等）。

**补丁说明：** MLX 内存环境变量、`mx.disable_compile()`、推理时手动覆盖 `model.names` 属于环境稳定性补丁，不属于消融变量，基线与实验 A 两组必须同时启用，否则对比无效。

# 7\. 本地源码修改记录（复现必看）

**区分两类修改：**

- **消融实验变量修改**：只改 data\.yaml 里的增强参数，不改源码；

- **本地源码补丁**：为了跑通训练所做的源码改动，属于全局稳定性补丁，基线 / 实验 A / 实验 B 三组都必须启用，否则对比不公平。

## 7\.1 调试用：coco\_dataset\.py 打印 Mosaic 开关

文件：`yolo26mlx/data/coco_dataset.py`。在增强初始化处加一行打印，用来确认 yaml 里的 mosaic 参数是否真的被读取：

```python
print(f"[DEBUG] Mosaic enabled: {self.mosaic}")
```

启动训练时观察控制台输出，确认 `mosaic: 0.0` 真的生效。

## 7\.2 实验 B 专用：loss\.py 关闭 one2many 分支

**实验 A 不需要改这一段！** 这是后面消融实验 B 才要改的代码，提前列出来防止混淆。

文件：`yolo26mlx/utils/loss.py`。找到 `E2ELoss` 里同时计算 one2one 与 one2many 两个分支 loss 的位置：

```python
loss_one2one, _ = compute_loss(preds_one2one, targets)
loss_one2many, _ = compute_loss(preds_one2many, targets)
total_loss = loss_one2one + loss_one2many
```

实验 B 改为只保留 one2one：

```python
loss_one2one, _ = compute_loss(preds_one2one, targets)
# loss_one2many, _ = compute_loss(preds_one2many, targets)  # 注释掉
total_loss = loss_one2one
```

前向仍输出两组头，但 one2many 不再回传梯度，推理评估仍取 one2one。

## 7\.3 用 git 回溯本地改动

在项目根目录执行以下命令，查看所有未提交的源码改动：

```bash
cd /Users/henry/git/AI-Projects/YOLO/yolo26-mlx
git diff                 # 查看改动
git status               # 查看修改过哪些文件
```

每做一组消融实验前，建议提交一个干净版本，方便切换：

```bash
git add .
git commit -m "stable: 实验A基线，MLX 内存补丁开启，one2many 分支开启"
git tag exp_A_baseline
```

**关键提醒：**实验 A 本身不需要修改模型 / 训练器源码；如果在调试期间临时改过其他文件，基线对照组必须使用同一版源码，保证两组代码完全一致。

# 8\. 实测参考数据（PCB SMD 检测）

以下为关闭 Mosaic 后，yolo26n \+ AdamW \+ batch=4 在 SMD 元器件数据集上的训练曲线，供调参参考：

|Epoch|Loss|mAP50|mAP50\-95|Recall|
|---|---|---|---|---|
|1|8\.97|0\.8206|0\.5695|0\.9815|
|5|6\.17|0\.9137|0\.6513|0\.9904|
|10|5\.46|0\.9407|0\.6913|0\.9936|
|15|4\.89|0\.9587|0\.7284|0\.9917|
|20|4\.55|0\.9668|0\.7462|0\.9962|
|25|4\.20|0\.9674|0\.7633|0\.9955|
|30|4\.04|**0\.9751**|**0\.7712**|0\.9943|

对比开启 Mosaic 的基线（mAP 从 0\.83 单边跌到 0\.03），关闭 Mosaic 后曲线单调上升，后期稳定在 0\.97 附近。

# 9\. 消融实验 B（预留）：关闭 one2many 分支

**实验目的：** 验证 one2many 多匹配分支是否是验证集指标震荡的根源。

**实验设计：** 在实验 A（mosaic=0）的基础上，只改一个变量——是否回传 one2many 分支 loss。

## 9\.1 变量定义

|条件|实验 A 组（对照）|实验 B 组|
|---|---|---|
|Mosaic / MixUp / CopyPaste|全部 0\.0（沿用实验 A）|全部 0\.0（沿用实验 A）|
|one2many loss|开启（total = o2o \+ o2m）|关闭（total = o2o）|
|其余所有条件|模型 / 数据 / LR / batch / 环境变量|与实验 A 完全一致|

## 9\.2 操作步骤

1. 基于实验 A 的 `smd-detect-no-mosaic.yaml` 复制一份 `smd-detect-no-o2m.yaml`，yaml 内容不改。

2. 按 7\.2 节修改 `yolo26mlx/utils/loss.py`，注释 one2many loss。

3. 启动训练，name 改为 `exp_B_1`，其余参数与实验 A 完全相同。

4. 训练结束后用 `git checkout` 还原 loss\.py，避免污染后续实验。

**注意：** 实验 B 的 yaml 增强参数、训练脚本、MLX 环境变量全部沿用实验 A 配置，保证单变量只在 one2many loss。每次跑完用 git 还原源码。

# 10\. 关键检查清单（复现实验前核对）

* [ ] Conda 环境已激活，MLX 版本 ≤ 0\.18\.x

* [ ] data\.yaml 中 nc、names、kpt\_shape 与标注一致

* [ ] mosaic / mixup 写在 data\.yaml，不是 train\(\) 参数

* [ ] 训练时 freeze\_bn=True（小 batch 必开）

* [ ] 推理前打印 model\.names 确认是自定义类别，不是 COCO

* [ ] Metal OOM 时先降 batch 到 2，再关内存池

* [ ] 基线 / 实验组源码版本一致，git diff 已确认无未预期改动

* [ ] 训练前已 export MLX\_DISABLE\_MEMORY\_POOL=1 与 MLX\_ALLOCATOR=system

* [ ] mx\.disable\_compile\(\) 两组统一开启或关闭，不能一组开一组关

* [ ] data\.yaml 中 path 使用绝对路径，外接卷 /Volumes 可访问

* [ ] 实验 B 跑完后已 git checkout 还原 loss\.py，避免污染后续实验

