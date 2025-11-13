# ========================================
# Fix ISL Dataset Structure
# Move labels from subfolders to correct location
# ========================================

import shutil
import os
from pathlib import Path

print("🔧 Fixing ISL dataset structure...")
print("="*60)

base_path = Path(r'D:\project')

# ========================================
# STEP 1: Move all labels to correct location
# ========================================
print("\n📁 Step 1: Moving labels from subfolders...")

train_labels_base = base_path / 'train' / 'labels'
val_labels_base = base_path / 'val' / 'labels'

# Find all .txt files in train/labels subfolders
train_label_files = list(train_labels_base.rglob('*.txt'))
print(f"Found {len(train_label_files)} label files in train/labels/")

# Move train labels to main labels folder
moved_train = 0
for label_file in train_label_files:
    if label_file.parent != train_labels_base:  # If in subfolder
        dest = train_labels_base / label_file.name
        if not dest.exists():
            shutil.copy2(label_file, dest)
            moved_train += 1

print(f"✅ Copied {moved_train} labels to train/labels/")

# Find all .txt files in val/labels subfolders
val_label_files = list(val_labels_base.rglob('*.txt'))
if len(val_label_files) > 0:
    moved_val = 0
    for label_file in val_label_files:
        if label_file.parent != val_labels_base:
            dest = val_labels_base / label_file.name
            if not dest.exists():
                shutil.copy2(label_file, dest)
                moved_val += 1
    print(f"✅ Copied {moved_val} labels to val/labels/")

# ========================================
# STEP 2: Move all images to correct location
# ========================================
print("\n📁 Step 2: Organizing images...")

train_images_base = base_path / 'train' / 'images'
val_images_base = base_path / 'val' / 'images'

# Find all images in train/images subfolders
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
train_image_files = []
for ext in image_extensions:
    train_image_files.extend(list(train_images_base.rglob(f'*{ext}')))

print(f"Found {len(train_image_files)} image files in train/images/")

# Move train images to main images folder
moved_train_imgs = 0
for img_file in train_image_files:
    if img_file.parent != train_images_base:  # If in subfolder
        dest = train_images_base / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)
            moved_train_imgs += 1

print(f"✅ Copied {moved_train_imgs} images to train/images/")

# Do the same for val
val_image_files = []
for ext in image_extensions:
    val_image_files.extend(list(val_images_base.rglob(f'*{ext}')))

if len(val_image_files) > 0:
    moved_val_imgs = 0
    for img_file in val_image_files:
        if img_file.parent != val_images_base:
            dest = val_images_base / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
                moved_val_imgs += 1
    print(f"✅ Copied {moved_val_imgs} images to val/images/")

# ========================================
# STEP 3: Split data into train/val
# ========================================
print("\n📁 Step 3: Creating train/val split (80/20)...")

import random

# Get all labels in train folder (now at correct level)
all_train_labels = list(train_labels_base.glob('*.txt'))
print(f"Total training labels: {len(all_train_labels)}")

# Calculate 20% for validation
val_count = int(len(all_train_labels) * 0.2)
val_count = max(val_count, 10)  # At least 10 validation samples

print(f"Moving {val_count} samples to validation...")

# Randomly select for validation
random.seed(42)
val_label_files = random.sample(all_train_labels, min(val_count, len(all_train_labels)))

# Move to validation
moved = 0
for label_file in val_label_files:
    # Find corresponding image
    img_name = label_file.stem
    
    img_file = None
    for ext in image_extensions:
        potential_img = train_images_base / f'{img_name}{ext}'
        if potential_img.exists():
            img_file = potential_img
            break
    
    if img_file and img_file.exists():
        try:
            # Move label
            dest_label = val_labels_base / label_file.name
            shutil.move(str(label_file), str(dest_label))
            
            # Move image
            dest_img = val_images_base / img_file.name
            shutil.move(str(img_file), str(dest_img))
            moved += 1
        except Exception as e:
            print(f"  Warning: {e}")

print(f"✅ Moved {moved} samples to validation")

# ========================================
# STEP 4: Update data.yaml
# ========================================
print("\n📁 Step 4: Updating data.yaml...")

yaml_content = f"""# ISL Dataset Configuration
path: {str(base_path)}
train: train/images
val: val/images

# Number of classes
nc: 36

# Class names
names:
  0: '0'
  1: '1'
  2: '2'
  3: '3'
  4: '4'
  5: '5'
  6: '6'
  7: '7'
  8: '8'
  9: '9'
  10: 'A'
  11: 'B'
  12: 'C'
  13: 'D'
  14: 'E'
  15: 'F'
  16: 'G'
  17: 'H'
  18: 'I'
  19: 'J'
  20: 'K'
  21: 'L'
  22: 'M'
  23: 'N'
  24: 'O'
  25: 'P'
  26: 'Q'
  27: 'R'
  28: 'S'
  29: 'T'
  30: 'U'
  31: 'V'
  32: 'W'
  33: 'X'
  34: 'Y'
  35: 'Z'
"""

with open(base_path / 'data.yaml', 'w') as f:
    f.write(yaml_content)

print("✅ data.yaml updated")

# ========================================
# STEP 5: Verify final structure
# ========================================
print("\n📊 Final dataset structure:")
print("="*60)

train_images_final = len(list(train_images_base.glob('*.[jJ][pP][gG]'))) + \
                     len(list(train_images_base.glob('*.[pP][nN][gG]')))
train_labels_final = len(list(train_labels_base.glob('*.txt')))
val_images_final = len(list(val_images_base.glob('*.[jJ][pP][gG]'))) + \
                   len(list(val_images_base.glob('*.[pP][nN][gG]')))
val_labels_final = len(list(val_labels_base.glob('*.txt')))

print(f"✅ Train images: {train_images_final}")
print(f"✅ Train labels: {train_labels_final}")
print(f"✅ Val images: {val_images_final}")
print(f"✅ Val labels: {val_labels_final}")

if train_labels_final > 0 and val_labels_final > 0:
    print("\n🎉 SUCCESS! Dataset is ready for training!")
    print("\nYou can now:")
    print("  1. Upload data.yaml to Colab")
    print("  2. Upload train/ and val/ folders to Colab")
    print("  3. Run the training script")
else:
    print("\n❌ Something went wrong. Check the folders manually.")

print("="*60)