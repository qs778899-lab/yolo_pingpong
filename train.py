from ultralytics import YOLO

# conda activate yolov11 && python train.py

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# 加载你之前的阶段性成果
# model = YOLO("runs/detect/train7/weights/epoch50.pt")

# Train the model
results = model.train(data="/home/s114/yolo/dataset/pingpong.yaml", epochs=500, imgsz=960, device=0, cache=False, plots=True, save_period=20, workers=4, batch=8, name="pingpong")

# results = model.train(data="/home/s114/yolo/dataset/pingpong.yaml", epochs=30, imgsz=640, device='cpu')