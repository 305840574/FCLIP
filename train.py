import argparse
import logging
import os
import os.path as osp
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
import segearth_segmentor
import custom_datasets
from mmseg.registry import RUNNERS
from Hook import SaveTrainableModulesHook



def parse_args():
    # 创建命令行参数解析器，描述为训练分割模型
    parser = argparse.ArgumentParser(description='SegEarth-OV Training with MMSeg')
    # 必填参数：配置文件路径
    parser.add_argument('--config', help='train config file path')
    # 可选参数：工作目录，用于保存日志和模型
    parser.add_argument('--work-dir', default='./train_logs/')
    # 可选参数：是否从最新检查点恢复训练
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    # 可选参数：是否启用自动混合精度训练
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    
    # 可选参数：分布式训练的启动器类型
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # 可选参数：本地进程排名，用于分布式训练，兼容 PyTorch >= 2.0.0
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    # 解析命令行参数
    args = parser.parse_args()
    # 如果环境变量中没有 LOCAL_RANK，则设置为命令行传入的值
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)
    # 设置启动器类型
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir


    # 如果启用自动混合精度训练（AMP）
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        # 检查优化器包装器是否已经启用 AMP
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            # 确保优化器包装器类型支持 AMP
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # 设置是否从检查点恢复训练
    #cfg.resume = args.resume
    
    runner = Runner.from_cfg(cfg)

    # 启动训练过程
    runner.train()

if __name__ == '__main__':
    # 脚本入口，运行 main 函数
    torch.autograd.set_detect_anomaly(True)
    main()