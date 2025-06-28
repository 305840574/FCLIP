from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS
import torch
import os


@HOOKS.register_module()
class SaveTrainableModulesHook(Hook):
    def __init__(self, interval=5, metric_key='mIoU', rule='greater'):
        self.interval = interval
        self.metric_key = metric_key
        self.rule = rule
        self.best_score = None
        self.best_ckpt_path = None

    def after_train_epoch(self, runner: Runner):
        epoch = runner.epoch

        if (epoch + 1) % self.interval != 0:
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        trainable_state_dict = {
            name: param for name, param in model.state_dict().items()
            if model.get_parameter(name).requires_grad
        }

        # 1. 保存当前 trainable 参数
        save_path = os.path.join(runner.work_dir, f'trainable_epoch_{epoch + 1}.pth')
        torch.save({'state_dict': trainable_state_dict}, save_path)
        runner.logger.info(f'[SaveTrainableModulesHook] ✅ 保存 trainable 权重到：{save_path}')

        # 2. 打印所有正在更新的模块名
        runner.logger.info('[SaveTrainableModulesHook] 📌 正在更新的参数（完整名称）：')
        for name, param in model.named_parameters():
            if param.requires_grad:
                runner.logger.info(f'  - {name}')

    def after_val_epoch(self, runner: Runner,**kwargs):
        """在验证阶段结束后检查是否需要保存最佳权重"""

        metrics = kwargs.get('metrics', None)

        if metrics is None:
            runner.logger.warning(f"[SaveTrainableModulesHook] 无法找到 metric：val/{self.metric_key}")
            return

        score = metrics.get(self.metric_key, None)
        if score is None:
            runner.logger.warning(f"[SaveTrainableModulesHook] 无法找到 metric：val/{self.metric_key}")
            return

        improved = (
            (self.best_score is None)
            or (self.rule == 'greater' and score > self.best_score)
            or (self.rule == 'less' and score < self.best_score)
        )

        if improved:
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            trainable_state_dict = {
                name: param for name, param in model.state_dict().items()
                if model.get_parameter(name).requires_grad
            }

            # 保存 best 权重
            best_path = os.path.join(runner.work_dir, f'best_{self.metric_key}.pth')
            torch.save({'state_dict': trainable_state_dict}, best_path)
            self.best_score = score
            self.best_ckpt_path = best_path
            runner.logger.info(f'[SaveTrainableModulesHook] 🏅 新 best {self.metric_key}={score:.4f}，权重已保存：{best_path}')
