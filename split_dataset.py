import os
import random
import shutil

dataset_path = "train_data"
output_path = "dataset"

train_ratio = 0.7
val_ratio = 0.15

classes = os.listdir(dataset_path)

for cls in classes:

    cls_path = os.path.join(dataset_path, cls)
    images = os.listdir(cls_path)

    random.shuffle(images)

    train_split = int(len(images) * train_ratio)
    val_split = int(len(images) * (train_ratio + val_ratio))

    train_imgs = images[:train_split]
    val_imgs = images[train_split:val_split]
    test_imgs = images[val_split:]

    for split, img_list in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):

        split_dir = os.path.join(output_path, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in img_list:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)