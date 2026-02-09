from ultralytics import YOLO

# Load a model

model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/s114/yolo/dataset/pingpong.yaml",, epochs=100, imgsz=640, device=0)