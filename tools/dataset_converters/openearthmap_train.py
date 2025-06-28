import argparse
import os
import os.path as osp
import shutil
from pathlib import Path

import numpy as np
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OpenEarthMap dataset to mmsegmentation format (train + val)')
    parser.add_argument('dataset_path', help='OpenEarthMap 根目录路径')
    parser.add_argument('-o', '--out_dir', help='输出目录', default=None)
    return parser.parse_args()


def collect_split(dataset_path, split, out_dir):
    """
    递归收集 dataset_path 下所有 images 下的 *.tif，
    根据 split.txt 里的文件名筛选，然后复制到 out_dir/img_dir/{split} 和 ann_dir/{split}
    """
    txt_path = osp.join(dataset_path, f'{split}.txt')
    names = np.loadtxt(txt_path, dtype=str).tolist()  # list of numpy.str_ → str
    
    # 找到所有 images 下的 tif 文件
    all_images = [
        p for p in Path(dataset_path).rglob("*.tif")
        if "/images/" in str(p)
    ]
    # 只保留在 names 列表中的那部分
    sel_images = [p for p in all_images if p.name in names]

    print(f'  {split}: 找到 {len(sel_images)} 张图 ({len(names)} 个文件名)')

    # 开始复制
    for p in sel_images:
        dst_img = osp.join(out_dir, 'img_dir', split, p.name)
        dst_ann = osp.join(out_dir, 'ann_dir', split, p.name)
        # 标签路径：把路径里的 images → labels
        ann_src = Path(str(p).replace('/images/', '/labels/'))
        mkdir_or_exist(osp.dirname(dst_img))
        mkdir_or_exist(osp.dirname(dst_ann))
        shutil.copy(str(p), dst_img)
        shutil.copy(str(ann_src), dst_ann)


def main():
    args = parse_args()
    dataset_path = args.dataset_path.rstrip('/')  # 去掉末尾的斜杠
    out_dir = args.out_dir or osp.join('data', 'OpenEarthMap')

    print('Making directories...')
    mkdir_or_exist(out_dir)
    for split in ['train', 'val']:
        mkdir_or_exist(osp.join(out_dir, 'img_dir', split))
        mkdir_or_exist(osp.join(out_dir, 'ann_dir', split))

    print(f'Processing dataset at {dataset_path} → {out_dir}')
    collect_split(dataset_path, 'train', out_dir)
    collect_split(dataset_path, 'val', out_dir)

    print('Done!')


if __name__ == '__main__':
    main()
