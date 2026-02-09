
from ultralytics import YOLO
# 1. 加载你训练好的模型
# 注意：'runs/detect/train/weights/best.pt' 是默认保存路径
# 如果你运行了多次训练，路径可能是 train2, train3 等，请根据实际文件夹名称修改
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 运行验证
# 验证过程会评估模型在验证集（val）上的表现
metrics = model.val(
    data="/home/s114/yolo/dataset/pingpong.yaml", # 必须指向你的数据集配置
    imgsz=640, 
    batch=16, 
    conf=0.25, # 置信度阈值
    iou=0.6,   # 交并比阈值，用于非极大值抑制（NMS）
    device=0,  # 使用 5060 显卡
    plots=True # 生成混淆矩阵、F1曲线等图表
)

# 3. 打印关键指标
print(f"mAP@50-95: {metrics.box.map:.4f}")    # 综合精度
print(f"mAP@50:    {metrics.box.map50:.4f}")  # 简单精度（常用指标）
print(f"精度 (P):   {metrics.box.mp:.4f}")     # Precision
print(f"召回率 (R): {metrics.box.mr:.4f}")     # Recall

# 4. 打印混淆矩阵（如果你想看具体的分类情况）
# 对于单类别（乒乓球）来说，主要看是否有漏检
print("\n混淆矩阵:")
print(metrics.confusion_matrix.matrix)