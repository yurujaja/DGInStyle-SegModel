# ---------------------------------------------------------------
# Author Yuru Jia. Last modified: 05/01/2024
# ---------------------------------------------------------------

import os
import os.path as osp
import math
import json
import numpy as np
import torch
from tqdm import tqdm
import shutil
from glob import glob


def copy_imgs(src_dir, tar_dir):
    files = sorted(glob(osp.join(src_dir, "images", "*.png")))
    for file in tqdm(files):
        base_name = osp.basename(file)
        new_basename = "n_" + base_name
        shutil.copy(file, osp.join(tar_dir, "images", new_basename))

        src_label = osp.join(src_dir, "labels", base_name.replace(".png", "_labelTrainIds.png"))

        tar_label = osp.join(tar_dir, "labels", new_basename.replace(".png", "_labelTrainIds.png"))
        shutil.copy(src_label, tar_label)


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


def rcs_gta_selection(rcs_data_root, 
                    output_folder, 
                    n_images=6000, 
                    rcs_temperature=0.01,
                    rcs_min_pixels=3000):
    # Set up RCS    
    rcs_classes, rcs_classprob = get_rcs_class_probs(rcs_data_root, rcs_temperature)
    with open(osp.join(rcs_data_root, 'samples_with_class.json'), 'r') as of:
        samples_with_class_and_n = json.load(of)
    
    samples_with_class_and_n = { int(k): v 
                                for k, v in samples_with_class_and_n.items() 
                                if int(k) in rcs_classes}
    
    samples_with_class = {}
    for c in rcs_classes:
        samples_with_class[c] = []
        for file, pixels in samples_with_class_and_n[c]:
            if pixels > rcs_min_pixels:
                samples_with_class[c].append(file)
        assert len(samples_with_class[c]) > 0


    selected_images = []
    selected_classes = []
    count = 0
    while True:        
        if count == n_images:
            break
        c = np.random.choice(rcs_classes, p=rcs_classprob)
        label_id_file = np.random.choice(samples_with_class[c])
        if label_id_file in selected_images:
            # print("Duplicated image!")
            continue
        selected_images.append(label_id_file)
        selected_classes.append(str(c))
        count += 1

    # Generate rcs selected files
    os.makedirs("rcs_sample_gta_files", exist_ok=True)
    rcs_gta_file = f'rcs_sample_gta_files/rcs_images_{n_images}.json'
    with open(rcs_gta_file, 'w', encoding='utf-8') as of:
        for file, c in zip(selected_images, selected_classes):
            of.write(f'{file},{c}\n')
    print('Finish selecting gta images!')

    # Copy images 
    
    
    with open(rcs_gta_file, 'r', encoding='utf-8') as of:
        lines = of.readlines()
    
    tar_folder = osp.join(output_folder,f"gta{n_images}_rcs1e{int(math.log10(rcs_temperature))}")
    os.makedirs(tar_folder, exist_ok=True)
    os.makedirs(osp.join(tar_folder,"images"), exist_ok=True)
    os.makedirs(osp.join(tar_folder,"labels"), exist_ok=True)
    print(f'Start copying selected gta images into {tar_folder}.')

    for line in tqdm(lines):
        file, c = line.strip().split(',')
        
        basename_label_src = osp.basename(file)        
        basename_label_tar = c + '_' + basename_label_src
        basename_img_tar = c + '_' + basename_label_src.replace('_labelTrainIds.png', '.png')   

        
        tar_image_path = osp.join(tar_folder,"images", basename_img_tar)
        tar_label_path = osp.join(tar_folder,"labels", basename_label_tar)

        src_img_path = file.replace('_labelTrainIds.png', '.png').replace('labels', 'images')
        src_label_path = file

        shutil.copy(src_img_path, tar_image_path)
        shutil.copy(src_label_path, tar_label_path)


if __name__ == "__main__":
    rcs_data_root='/cluster/work/igp_psr/yuru/udadatasets/gta'
    output_folder=f'../data/'

    rcs_gta_selection(rcs_data_root, output_folder)




    