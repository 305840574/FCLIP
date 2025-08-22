import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import shutil
from os.path import join

# === 目录配置 ===
DATA_ROOT = "/root/autodl-tmp/zdj-SegEarth-OV/data/Million-AID/test"
BAD_IMAGE_DIR = "/root/autodl-tmp/zdj-SegEarth-OV/data/Million-AID/bad_images"

SAVE_BAD_IMAGES = True    # 保存异常图像副本
DELETE_BAD_IMAGES = True  # 删除原始异常图像

# === 数据集定义 ===
class MillionAIDDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.image_files = [join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        try:
            image = Image.open(path).convert("RGB")
            return {"img": self.transform(image), "img_path": path}
        except Exception as e:
            print(f"[错误] 无法加载图像 {path}：{e}")
            return {"img": None, "img_path": path}

# === 预处理变换 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = MillionAIDDataset(DATA_ROOT, transform)

means, stds = [], []
bad_images = []

# === 遍历图像，统计并筛选异常 ===
for i in range(len(dataset)):
    sample = dataset[i]
    img = sample["img"]
    img_path = sample["img_path"]

    if img is None:
        bad_images.append(img_path)
        continue

    mean = img.mean().item()
    std = img.std().item()
    means.append(mean)
    stds.append(std)

    if mean < 0.1 or mean > 0.9 or std < 0.05:
        print(f"[异常] {img_path} | mean={mean:.4f}, std={std:.4f}")
        bad_images.append(img_path)

        if SAVE_BAD_IMAGES:
            os.makedirs(BAD_IMAGE_DIR, exist_ok=True)
            shutil.copy(img_path, join(BAD_IMAGE_DIR, os.path.basename(img_path)))

# === 删除原始异常图像 ===
if DELETE_BAD_IMAGES:
    for bad_img in bad_images:
        try:
            os.remove(bad_img)
            print(f"[已删除] {bad_img}")
        except Exception as e:
            print(f"[删除失败] {bad_img}：{e}")

# === 可视化分布 ===
plt.hist(means, bins=50)
plt.title("Mean pixel value per image")
plt.xlabel("Mean")
plt.ylabel("Image Count")
plt.grid(True)
plt.show()

plt.figure()
plt.hist(stds, bins=50)
plt.title("STD pixel value per image")
plt.xlabel("STD")
plt.ylabel("Image Count")
plt.grid(True)
plt.show()

# === 打印统计信息 ===
print("\n========== 统计信息 ==========")
print(f"总图像数         ：{len(means) + len(bad_images)}")
print(f"异常图像数（总） ：{len(bad_images)}")
print(f"平均 mean        ：{np.mean(means):.4f}")
print(f"平均 std         ：{np.mean(stds):.4f}")
