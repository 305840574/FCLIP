import torch

# 加载两个 ckpt 文件
ckpt1 = torch.load("/root/autodl-tmp/zdj-SegEarth-OV/work_dirs/simfeatup_million_aid/checkpoints/jbu_one/fusion/xclip_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_2000.ckpt")
ckpt2 = torch.load("/root/autodl-tmp/zdj-SegEarth-OV/work_dirs/simfeatup_million_aid/checkpoints/jbu_one/fusion/xclip_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_20000.ckpt")

# 获取 state_dict
state_dict1 = ckpt1.get('state_dict', ckpt1)
state_dict2 = ckpt2.get('state_dict', ckpt2)

# 比较键
keys1 = set(state_dict1.keys())
keys2 = set(state_dict2.keys())

print("Keys only in state_dict1:", keys1 - keys2)
print("Keys only in state_dict2:", keys2 - keys1)
print("Common keys:", keys1 & keys2)

# 比较权重值
differences = []
for name in keys1 & keys2:  # 只比较共同的键
    tensor1 = state_dict1[name]
    tensor2 = state_dict2[name]
    
    # 检查形状是否一致
    if tensor1.shape != tensor2.shape:
        differences.append(f"Parameter {name} has different shapes: {tensor1.shape} vs {tensor2.shape}")
        continue
    
    # 检查值是否一致
    if not torch.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8):
        diff = (tensor1 - tensor2).abs().max().item()  # 最大绝对差
        differences.append(f"Parameter {name} differs (max diff: {diff:.6f})")

# 输出比较结果
if differences:
    print("Found differences between state_dict1 and state_dict2:")
    for diff in differences:
        print(diff)
else:
    print("No differences found between state_dict1 and state_dict2.")