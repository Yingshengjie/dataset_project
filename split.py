# -*- coding: utf-8 -*-
import os, random, shutil

img_dir = r"D:\all_pics"
txt_dir = r"D:\all_labels"
out_root = r"D:\my_dataset"
train_rate = 0.7

os.makedirs(os.path.join(out_root,"images/train"), exist_ok=True)
os.makedirs(os.path.join(out_root,"images/val"), exist_ok=True)
os.makedirs(os.path.join(out_root,"labels/train"), exist_ok=True)
os.makedirs(os.path.join(out_root,"labels/val"), exist_ok=True)

img_list = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
print(f"number:{len(img_list)}")
print(img_list[:5])

random.shuffle(img_list)
split_num = int(len(img_list)*train_rate)

for i, fname in enumerate(img_list):
    base, ext = os.path.splitext(fname)
    txt_name = base + ".txt"
    src_img = os.path.join(img_dir, fname)
    src_txt = os.path.join(txt_dir, txt_name)
    if i < split_num:
        shutil.copy(src_img, os.path.join(out_root,"images/train",fname))
        if os.path.exists(src_txt):
            shutil.copy(src_txt, os.path.join(out_root,"labels/train",txt_name))
    else:
        shutil.copy(src_img, os.path.join(out_root,"images/val",fname))
        if os.path.exists(src_txt):
            shutil.copy(src_txt, os.path.join(out_root,"labels/val",txt_name))

print("done")
input("press enter")