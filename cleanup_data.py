import os
from pathlib import Path

# conda activate yolov11 && python cleanup_data.py

def cleanup_dataset(data_dir):
    data_path = Path(data_dir)
    # 定义需要检查的子目录对 (train 和 val)
    subsets = ['train', 'val']
    
    for subset in subsets:
        print(f"\n正在清理 {subset} 子集...")
        img_dir = data_path / 'images' / subset
        lab_dir = data_path / 'labels' / subset
        
        if not img_dir.exists() or not lab_dir.exists():
            print(f"警告: 路径不存在，跳过 {subset}")
            continue

        # 获取所有图片和标签的文件名（不含后缀）
        # 支持多种常见图片格式
        valid_img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        img_files = {f.stem: f.suffix for f in img_dir.iterdir() if f.suffix.lower() in valid_img_extensions}
        lab_files = {f.stem: f.suffix for f in lab_dir.iterdir() if f.suffix.lower() == '.txt'}
        
        img_names = set(img_files.keys())
        lab_names = set(lab_files.keys())
        
        # 1. 删除没有标签的图片 (背景图)
        imgs_to_delete = img_names - lab_names
        for img_name in imgs_to_delete:
            ext = img_files[img_name]
            img_path = img_dir / f"{img_name}{ext}"
            try:
                os.remove(img_path)
                print(f"  [删除] 无标签图片: {img_path.name}")
            except Exception as e:
                print(f"  [错误] 无法删除图片 {img_name}: {e}")
                
        # 2. 删除没有图片的标签 (孤立标签)
        labs_to_delete = lab_names - img_names
        for lab_name in labs_to_delete:
            ext = lab_files[lab_name]
            lab_path = lab_dir / f"{lab_name}{ext}"
            try:
                os.remove(lab_path)
                print(f"  [删除] 无图片标签: {lab_path.name}")
            except Exception as e:
                print(f"  [错误] 无法删除标签 {lab_name}: {e}")

        if not imgs_to_delete and not labs_to_delete:
            print(f"  {subset} 子集已经很干净了，无需删除。")

if __name__ == "__main__":
    # 指定你的数据集根目录
    dataset_root = "/home/s114/yolo/dataset"
    cleanup_dataset(dataset_root)
    print("\n数据集清理完成！")


'''
找到标注有问题的数据：单张图重复标注
cd /home/s114/yolo/dataset/labels/train
for f in *.txt; do
  n=$(wc -l < "$f")
  if [ "$n" -gt 1 ]; then
    echo "$f 有 $n 行(个框)"
  fi
done

cd /home/s114/yolo/dataset/labels/val
for f in *.txt; do
  n=$(wc -l < "$f")
  if [ "$n" -gt 1 ]; then
    echo "$f 有 $n 行(个框)"
  fi
done


'''