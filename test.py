import glob
import numpy as np
from PIL import Image
from collections import Counter

mask = np.array(Image.open("/root/autodl-tmp/zdj-SegEarth-OV/data/UAVid/ann_dir/test/seq16_000000_0_1080_0_1280.png"))
print(np.unique(mask))
'''
import argparse
import glob
import os.path as osp
import numpy as np
from PIL import Image
import mmcv

# iSAID调色板
iSAID_palette = {
    0: (0, 0, 0), 1: (0, 0, 63), 2: (0, 63, 63), 3: (0, 63, 0),
    4: (0, 63, 127), 5: (0, 63, 191), 6: (0, 63, 255), 7: (0, 127, 63),
    8: (0, 127, 127), 9: (0, 0, 127), 10: (0, 0, 191), 11: (0, 0, 255),
    12: (0, 191, 127), 13: (0, 127, 191), 14: (0, 127, 255), 15: (0, 100, 155)
}

# 将调色板转换为NumPy数组以便高效比较
PALETTE_COLORS = np.array(list(iSAID_palette.values()))
PALETTE_INDICES = np.array(list(iSAID_palette.keys()))

def verify_label_image(image_path):
    """
    验证单张iSAID标签图像的颜色值。
    """
    if not osp.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"\n--- Verifying: {osp.basename(image_path)} ---")

    # 方法1: 使用 PIL (Image.open)
    try:
        pil_img = Image.open(image_path).convert('RGB')
        pil_arr = np.array(pil_img)
        print("  - Loaded with PIL (Image.open)")
        verify_colors(pil_arr)
    except Exception as e:
        print(f"  - Error loading with PIL: {e}")

    print("-" * 20)

    # 方法2: 使用 mmcv.imread
    try:
        mmcv_arr = mmcv.imread(image_path, channel_order='rgb')
        print("  - Loaded with mmcv.imread")
        verify_colors(mmcv_arr)
    except Exception as e:
        print(f"  - Error loading with mmcv: {e}")

def verify_colors(image_array):
    """
    核心函数：比较图像中的颜色与iSAID调色板。
    """
    if image_array is None:
        return

    # 获取图像中的所有唯一颜色
    unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)

    print(f"    - Found {len(unique_colors)} unique colors.")
    print("    - Checking for colors not in iSAID_palette...")

    unmatched_colors = []
    num_unmatched_pixels = 0
    total_pixels = image_array.shape[0] * image_array.shape[1]

    # 遍历图像中的所有像素，检查是否与调色板完全匹配
    arr_2d = np.zeros(image_array.shape[:2], dtype=np.uint8)
    matched_pixels_count = 0
    for color in unique_colors:
        # 使用np.any和np.all来检查颜色是否在调色板中
        is_in_palette = np.any(np.all(PALETTE_COLORS == color, axis=1))
        
        if not is_in_palette:
            unmatched_colors.append(color)
            
            # 计算不匹配的像素数量
            m = np.all(image_array == color, axis=2)
            num_unmatched_pixels += np.sum(m)
        else:
            m = np.all(image_array == color, axis=2)
            matched_pixels_count += np.sum(m)
    
    print(f"    - Total pixels: {total_pixels}")
    print(f"    - Pixels with colors from the palette: {matched_pixels_count}")
    print(f"    - **Pixels with colors NOT in the palette**: {num_unmatched_pixels}")
    
    if unmatched_colors:
        print("    - The following colors do not match any entry in iSAID_palette:")
        for color in unmatched_colors:
            print(f"      -> {color}")
        print("\n    Summary: The presence of unmatched colors and pixels confirms that "
              "a direct, strict RGB-to-index mapping (like in your original code) "
              "will fail to classify these pixels, leaving them as the default value (0).")
    else:
        print("    - All colors in this image are present in the iSAID_palette. "
              "This suggests the image might not have anti-aliasing artifacts or has been "
              "processed to remove them.")

def main():
    parser = argparse.ArgumentParser(description="Verify colors in iSAID label images.")
    parser.add_argument('label_path', help="Path to a single label image or a folder containing them.")
    args = parser.parse_args()

    label_path = args.label_path
    
    if osp.isdir(label_path):
        print(f"Searching for .png files in folder: {label_path}")
        image_files = glob.glob(osp.join(label_path, '*.png'))
        if not image_files:
            print("No .png files found.")
            return
        
        for img_file in image_files:
            verify_label_image(img_file)
    else:
        verify_label_image(label_path)

if __name__ == '__main__':
    main()
'''
