import os
configs_list = [
    # rs semantic segmentation
    './configs/cfg_openearthmap.py',
    './configs/cfg_loveda.py',
    './configs/cfg_vdd.py',
    './configs/cfg_potsdam.py',
    './configs/cfg_vaihingen.py',
    './configs/cfg_udd5.py',
]

for config in configs_list:
    print(f"Running {config}")
    # os.system(f"bash ./dist_test.sh {config}")
    os.system(f'python eval.py --config {config} --work-dir work_dirs/tmp')
