from ultralytics import YOLO
import torch 
import os
# 限制 PyTorch 内部计算使用的 CPU 线程数 (建议设为 2 或 4)
torch.set_num_threads(6)
# 限制底层库 (如 OpenBLAS, MKL) 使用的线程数
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

# conda activate yolov11 && python train.py

model = YOLO("yolo11n.pt")  # load a pretrained model
# 加载你之前的阶段性成果
# model = YOLO("runs/detect/pingpong2/weights/epoch50.pt")
# Train the model
results = model.train(data="/home/s114/yolo/dataset/pingpong.yaml", epochs=1500, imgsz=960, device=0, cache=False, plots=True, save_period=20, workers=4, batch=8, name="pingpong")
# workers影响CPU，batch影响GPU。
# results = model.train(data="/home/s114/yolo/dataset/pingpong.yaml", epochs=30, imgsz=640, device='cpu')


'''
#查找best.pt是第几轮epoch的结果

cd /home/s114/yolo && python3 -c "
import csv
with open('runs/detect/pingpong2/results.csv') as f:
    r = csv.DictReader(f)
    rows = list(r)
    col = 'metrics/mAP50-95(B)'
    best = max(rows, key=lambda x: float(x[col]))
    epoch = int(float(best['epoch']))
    m = float(best[col])
    print('best.pt 对应的是第', epoch, '轮 (epoch)，该轮 mAP50-95 =', round(m, 4))
"
'''

'''
查看当前conda环境的cuda版本：
python -c "import torch; print(torch.version.cuda)"

12.8
'''