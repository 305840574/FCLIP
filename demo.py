from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
import numpy as np

img = Image.open('/root/autodl-tmp/zdj-SegEarth-OV/data/iSAID/img_dir/val/P2689_1024_1920_2560_3456.png')

name_list=['background','ship','store tank','baseball diamond','tennis court','basketball court',
           'ground track field','bridge','large vehicle','small vehicle','helicopter','swimming pool',
           'roundabout','soccer ball field','plane','harbor']
#name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
#             'tree,forest', 'water,river', 'cropland', 'building,roof,house']
#name_list = ['background', 'building,roof,house', 'road', 'water', 'barren',
#             'forest', 'agricultural']
#name_list=['background','building']
with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = SegEarthSegmentation(
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'NACLIP'
    ignore_residual=True,
    isFusion=False,
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.3,
    feature_cls_token_lambda= 0,
    name_path='./configs/my_name.txt',
    prob_thd=0.1,
)

seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)
#np.savetxt('output.txt',seg_pred, fmt='%.1f')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis')
ax[1].axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('seg_pred_limit_orgin3', bbox_inches='tight')