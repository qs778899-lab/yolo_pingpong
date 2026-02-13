from ultralytics import YOLO
# conda activate yolov11 && python test.py
model = YOLO("runs/detect/pingpong2/weights/best.pt")
model.predict(source="/home/s114/yolo/dataset/images/val/frame_0_20260209_070628_936195_R.jpg", conf=0.8, show=True, save=True)

#当数据量极小时，conf要设置比较小，比如0.012