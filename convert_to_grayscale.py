import os
from PIL import Image
from pathlib import Path

# conda activate yolov11 && python convert_to_grayscale.py

def convert_images_to_grayscale(src_root, dst_root):
    """
    将 src_root 下的所有图片转换为灰度图，并保存到 dst_root。
    保持目录结构和文件名不变。
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # 支持的图片扩展名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # 遍历 train 和 val 文件夹
    for split in ['train', 'val']:
        src_dir = src_root / split
        dst_dir = dst_root / split

        if not src_dir.exists():
            print(f"Warning: 源目录 {src_dir} 不存在，跳过。")
            continue

        # 创建目标目录
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"正在处理 {split} 目录...")

        # 遍历目录下的所有文件
        for img_path in src_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                try:
                    # 打开图片并转换为灰度 (L 模式)
                    with Image.open(img_path) as img:
                        gray_img = img.convert('L')
                        
                        # 保存到新路径，保持文件名不变
                        save_path = dst_dir / img_path.name
                        gray_img.save(save_path)
                        # print(f"已保存: {save_path}")
                except Exception as e:
                    print(f"处理图片 {img_path} 时出错: {e}")

    print("所有图片转换完成！")

if __name__ == "__main__":
    # 源图片根目录
    source_images = "dataset/images"
    # 目标图片根目录
    target_images = "dataset/images2"

    convert_images_to_grayscale(source_images, target_images)
