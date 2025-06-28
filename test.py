import json
import matplotlib.pyplot as plt

json_path = '/data/SegEarth-OV/state_dict/VDD/no_class_weight_agjust/vis_data(1)/20250622_230124.json'

with open(json_path, 'r') as f:
    logs = []
    for line in f:
        try:
            log = json.loads(line)
            if 'iter' in log and 'loss' in log:  # 只保留包含 iter 和 loss 的项
                logs.append(log)
        except json.JSONDecodeError:
            continue  # 跳过格式有问题的行

iters = [log['iter'] for log in logs]
losses = [log['loss_ce'] for log in logs]
plt.plot(iters, losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(f'Loss Curve from {json_path}')
plt.grid(True)
plt.savefig('/data/SegEarth-OV/train_logs/20250616_230235/result.png')  # 保存图像
plt.show()
