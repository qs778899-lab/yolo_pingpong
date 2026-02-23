from ultralytics import YOLO
import shutil
import os

# conda activate yolov11  &&  python export_onnx.py

# 1. 加载训练好的 PyTorch 模型 (.pt)
model_path = "runs/detect/pingpong7/weights/best.pt"
model = YOLO(model_path)

# 2. 导出为 ONNX 格式
# export 会返回导出的文件路径
onnx_path = model.export(format="onnx", dynamic=True, opset=12)

# 3. 将导出文件移动到项目根目录下的 weights 文件夹
target_dir = "weights"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

target_path = os.path.join(target_dir, "best.onnx")

# 移动文件
if onnx_path and os.path.exists(onnx_path):
    shutil.move(onnx_path, target_path)
    print(f"模型已成功导出并移动到: {target_path}")
else:
    print("导出失败，请检查模型路径或环境。")
