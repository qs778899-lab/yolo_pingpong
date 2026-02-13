"""从 YOLO 训练产生的 results.csv 中找出 best.pt 对应的 epoch。"""
import csv
import sys

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/pingpong2/results.csv"
    try:
        with open(csv_path) as f:
            r = csv.DictReader(f)
            rows = list(r)
    except FileNotFoundError:
        print(f"文件不存在: {csv_path}")
        return
    col = "metrics/mAP50-95(B)"
    if col not in rows[0]:
        print("CSV 中未找到 mAP50-95 列")
        return
    best = max(rows, key=lambda x: float(x[col]))
    epoch = int(float(best["epoch"]))
    m = float(best[col])
    print(f"best.pt 对应的是第 {epoch} 轮 (epoch)，该轮 mAP50-95 = {m:.4f}")

if __name__ == "__main__":
    main()
