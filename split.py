import os, shutil, random

source_folder = r"D:\project\data"
train_folder = r"D:\project\train\images"
val_folder = r"D:\project\val\images"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

for cls in os.listdir(source_folder):
    class_path = os.path.join(source_folder, cls)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images)
    split_idx = int(0.8 * len(images))

    for i, img_name in enumerate(images):
        src = os.path.join(class_path, img_name)
        dest_folder = train_folder if i < split_idx else val_folder
        cls_folder = os.path.join(dest_folder, cls)
        os.makedirs(cls_folder, exist_ok=True)
        shutil.copy(src, os.path.join(cls_folder, img_name))

print("✅ Dataset split complete!")
