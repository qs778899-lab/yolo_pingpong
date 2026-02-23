import os
import shutil
from PIL import Image
from pathlib import Path

# conda activate yolov11 && python convert_to_grayscale.py

def convert_images_to_grayscale(src_images_root, dst_images_root, src_labels_root, dst_labels_root):
    """
    将 src_images_root 下的所有图片转换为灰度图，并保存到 dst_images_root。
    同时复制对应的标签文件到 dst_labels_root。
    保持目录结构和文件名不变。
    """
    src_images_root = Path(src_images_root)
    dst_images_root = Path(dst_images_root)
    src_labels_root = Path(src_labels_root)
    dst_labels_root = Path(dst_labels_root)

    # 支持的图片扩展名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # 遍历 train 和 val 文件夹
    for split in ['train', 'val']:
        src_img_dir = src_images_root / split
        dst_img_dir = dst_images_root / split
        src_lbl_dir = src_labels_root / split
        dst_lbl_dir = dst_labels_root / split

        if not src_img_dir.exists():
            print(f"Warning: 源图片目录 {src_img_dir} 不存在，跳过。")
            continue

        # 创建目标目录
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        print(f"正在处理 {split} 目录...")

        # 遍历目录下的所有文件
        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                try:
                    # 打开图片并转换为灰度，然后转回 RGB（保持 3 通道）
                    with Image.open(img_path) as img:
                        gray_img = img.convert('L')  # 转为灰度
                        rgb_gray_img = gray_img.convert('RGB')  # 转回 RGB 3 通道
                        
                        # 保存图片到新路径，保持文件名不变
                        save_img_path = dst_img_dir / img_path.name
                        rgb_gray_img.save(save_img_path)
                        
                        # 复制对应的标签文件
                        label_name = img_path.stem + '.txt'
                        src_label_path = src_lbl_dir / label_name
                        dst_label_path = dst_lbl_dir / label_name
                        
                        if src_label_path.exists():
                            shutil.copy2(src_label_path, dst_label_path)
                        else:
                            print(f"Warning: 找不到对应的标签文件 {src_label_path}")
                        
                except Exception as e:
                    print(f"处理图片 {img_path} 时出错: {e}")

    print("所有图片转换和标签复制完成！")

if __name__ == "__main__":
    # 源图片根目录
    source_images = "dataset/images"
    # 目标图片根目录
    target_images = "dataset/images2"
    # 源标签根目录
    source_labels = "dataset/labels"
    # 目标标签根目录
    target_labels = "dataset/labels2"

    convert_images_to_grayscale(source_images, target_images, source_labels, target_labels)
