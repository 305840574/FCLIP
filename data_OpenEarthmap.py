import os
import shutil
from pathlib import Path

def filter_xbd(csv_path, dataset_path, out_dir):
    # 读取 xBD 的 CSV 文件，提取 oem_name（OpenEarthMap 文件名）
    xbd_files = set()
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")  # 改为逗号分隔
            if len(parts) == 2:
                xbd_files.add(parts[1].strip())  # 去掉可能的空格
                print(parts[1])

    # 输入/输出路径
    img_dir = Path(dataset_path) / "img_dir" / "val"
    ann_dir = Path(dataset_path) / "ann_dir" / "val"
    out_img_dir = Path(out_dir) / "img_dir" / "val"
    out_ann_dir = Path(out_dir) / "ann_dir" / "val"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    kept, removed = 0, 0
    for img_file in img_dir.glob("*.tif"):
        fname = img_file.name
        if fname in xbd_files:
            removed += 1
            continue  # 跳过 xBD 文件
        # 拷贝图像和标注
        shutil.copy(img_file, out_img_dir / fname)
        ann_file = ann_dir / fname
        if ann_file.exists():
            shutil.copy(ann_file, out_ann_dir / fname)
        kept += 1

    print(f"✅ 完成！保留 {kept} 张非-xBD 验证图像，移除 {removed} 张 xBD 图像。")


if __name__ == "__main__":
    filter_xbd(
        csv_path="/root/autodl-tmp/zdj-SegEarth-OV/xbd_files.csv",
        dataset_path="/root/autodl-tmp/zdj-SegEarth-OV/data/OpenEarthMap",
        out_dir="/root/autodl-tmp/zdj-SegEarth-OV/data/OpenEarthMap_wo_xBD"
    )
