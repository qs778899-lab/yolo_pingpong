from ultralytics import YOLO

# conda activate yolov11
# python train.py

# Load a model

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/s114/yolo/dataset/pingpong.yaml", epochs=300, imgsz=1280, device=0, cache=True, plots=True)

# results = model.train(data="/home/s114/yolo/dataset/pingpong.yaml", epochs=30, imgsz=640, device='cpu')