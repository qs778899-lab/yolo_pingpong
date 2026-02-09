from ultralytics import YOLO
# conda activate yolov11 && python test.py
model = YOLO("runs/detect/train5/weights/best.pt")
model.predict(source="/home/s114/yolo/dataset/images/train/1a9a901d-frame_1_20260206_044214_847114_R.jpg", conf=0.012, show=True, save=True)

#当数据量极小时，conf要设置比较小，比如0.012