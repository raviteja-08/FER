import os
import shutil
import random

# Paths
source_dir = "N:/Major Project/fer_ck_kdef/fer_ckplus_kdef"
train_dir = "N:/Major Project/dataset/train"
test_dir = "N:/Major Project/dataset/test"
valid_dir = "N:/Major Project/dataset/valid"

# Split ratios
train_ratio = 0.75
test_ratio = 0.10
valid_ratio = 0.15

# Ensure output folders exist
for dir_path in [train_dir, test_dir, valid_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Process each emotion class
for emotion in os.listdir(source_dir):
    emotion_path = os.path.join(source_dir, emotion)
    if not os.path.isdir(emotion_path):
        continue  # skip non-directories

    images = os.listdir(emotion_path)
    random.shuffle(images)

    total = len(images)
    train_split = int(total * train_ratio)
    test_split = int(total * (train_ratio + test_ratio))

    train_images = images[:train_split]
    test_images = images[train_split:test_split]
    valid_images = images[test_split:]

    # Make subfolders for the emotion in each destination folder
    for folder, img_list in zip([train_dir, test_dir, valid_dir],
                                [train_images, test_images, valid_images]):
        emotion_dest = os.path.join(folder, emotion)
        os.makedirs(emotion_dest, exist_ok=True)
        for img in img_list:
            shutil.copy(os.path.join(emotion_path, img), os.path.join(emotion_dest, img))

    print(f"✅ {emotion}: {len(train_images)} train, {len(test_images)} test, {len(valid_images)} valid")

print("🎉 Split complete!")
