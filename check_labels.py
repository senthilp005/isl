# ========================================
# ISL (Indian Sign Language) Detection with YOLOv8
# Complete Training Pipeline for Google Colab
# ========================================

# STEP 1: Setup Environment
# ========================================
print("📦 Installing dependencies...")
!pip install -q ultralytics==8.3.227
!pip install -q opencv-python matplotlib tqdm

import os
import shutil
import yaml
from pathlib import Path
import torch
from IPython.display import Image, display

# Check GPU availability
print(f"\n🖥️ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ No GPU detected! Training will be slow.")

# STEP 2: Clone Dataset
# ========================================
print("\n📥 Cloning ISL dataset from GitHub...")
if os.path.exists('isl'):
    shutil.rmtree('isl')
!git clone https://github.com/senthilp005/isl.git
%cd isl

# STEP 3: Fix data.yaml with correct paths
# ========================================
print("\n🔧 Fixing data.yaml configuration...")

# Read the original data.yaml
with open('data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Get current working directory
base_path = os.getcwd()

# Update paths to absolute paths in Colab
data_config['path'] = base_path
data_config['train'] = 'train/images'
data_config['val'] = 'val/images'

# Ensure nc (number of classes) and names are set
if 'nc' not in data_config or 'names' not in data_config:
    # Count classes from train/labels
    label_files = list(Path('train/labels').glob('*.txt'))
    classes = set()
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                classes.add(class_id)
    
    data_config['nc'] = len(classes)
    data_config['names'] = [f'class_{i}' for i in range(len(classes))]

print(f"Dataset configuration:")
print(f"  - Classes: {data_config['nc']}")
print(f"  - Train path: {os.path.join(base_path, data_config['train'])}")
print(f"  - Val path: {os.path.join(base_path, data_config['val'])}")

# Save corrected data.yaml
with open('data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

# Verify dataset structure
print("\n📁 Verifying dataset structure...")
train_images = len(list(Path('train/images').glob('*')))
train_labels = len(list(Path('train/labels').glob('*.txt')))
val_images = len(list(Path('val/images').glob('*')))
val_labels = len(list(Path('val/labels').glob('*.txt')))

print(f"  ✓ Train images: {train_images}")
print(f"  ✓ Train labels: {train_labels}")
print(f"  ✓ Val images: {val_images}")
print(f"  ✓ Val labels: {val_labels}")

if train_images == 0 or train_labels == 0:
    print("  ❌ ERROR: No training data found!")
else:
    print("  ✅ Dataset structure looks good!")

# STEP 4: Configure Training Parameters
# ========================================
print("\n⚙️ Configuring training parameters for optimal 1-hour training...")

# Training configuration for ~1 hour on T4 GPU
TRAINING_CONFIG = {
    'model': 'yolov8n.pt',  # Nano model - fastest training
    'data': 'data.yaml',
    'epochs': 100,  # Will use early stopping
    'patience': 15,  # Early stopping patience
    'batch': 16,  # Adjust based on GPU memory
    'imgsz': 640,  # Standard image size
    'device': 0,  # GPU device
    'workers': 8,
    'project': 'runs/detect',
    'name': 'isl_model',
    'exist_ok': True,
    'pretrained': True,
    'optimizer': 'auto',
    'verbose': True,
    'seed': 0,
    'deterministic': True,
    'single_cls': False,
    'rect': False,
    'cos_lr': False,
    'close_mosaic': 10,
    'resume': False,
    'amp': True,  # Automatic Mixed Precision for faster training
    'fraction': 1.0,
    'profile': False,
    'freeze': None,
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'plots': True,
    'save': True,
    'save_period': -1,
    'cache': False,  # Set to 'ram' if you have enough memory
    'val': True,
}

print("Training Configuration:")
for key, value in TRAINING_CONFIG.items():
    print(f"  {key}: {value}")

# STEP 5: Train the Model
# ========================================
print("\n🚀 Starting YOLOv8 training...")
print("=" * 60)

from ultralytics import YOLO

# Initialize model
model = YOLO(TRAINING_CONFIG['model'])

# Train the model
results = model.train(
    data=TRAINING_CONFIG['data'],
    epochs=TRAINING_CONFIG['epochs'],
    patience=TRAINING_CONFIG['patience'],
    batch=TRAINING_CONFIG['batch'],
    imgsz=TRAINING_CONFIG['imgsz'],
    device=TRAINING_CONFIG['device'],
    workers=TRAINING_CONFIG['workers'],
    project=TRAINING_CONFIG['project'],
    name=TRAINING_CONFIG['name'],
    exist_ok=TRAINING_CONFIG['exist_ok'],
    pretrained=TRAINING_CONFIG['pretrained'],
    optimizer=TRAINING_CONFIG['optimizer'],
    verbose=TRAINING_CONFIG['verbose'],
    seed=TRAINING_CONFIG['seed'],
    deterministic=TRAINING_CONFIG['deterministic'],
    single_cls=TRAINING_CONFIG['single_cls'],
    rect=TRAINING_CONFIG['rect'],
    cos_lr=TRAINING_CONFIG['cos_lr'],
    close_mosaic=TRAINING_CONFIG['close_mosaic'],
    resume=TRAINING_CONFIG['resume'],
    amp=TRAINING_CONFIG['amp'],
    fraction=TRAINING_CONFIG['fraction'],
    profile=TRAINING_CONFIG['profile'],
    freeze=TRAINING_CONFIG['freeze'],
    lr0=TRAINING_CONFIG['lr0'],
    lrf=TRAINING_CONFIG['lrf'],
    momentum=TRAINING_CONFIG['momentum'],
    weight_decay=TRAINING_CONFIG['weight_decay'],
    warmup_epochs=TRAINING_CONFIG['warmup_epochs'],
    warmup_momentum=TRAINING_CONFIG['warmup_momentum'],
    warmup_bias_lr=TRAINING_CONFIG['warmup_bias_lr'],
    box=TRAINING_CONFIG['box'],
    cls=TRAINING_CONFIG['cls'],
    dfl=TRAINING_CONFIG['dfl'],
    plots=TRAINING_CONFIG['plots'],
    save=TRAINING_CONFIG['save'],
    save_period=TRAINING_CONFIG['save_period'],
    cache=TRAINING_CONFIG['cache'],
    val=TRAINING_CONFIG['val'],
)

print("\n✅ Training completed!")

# STEP 6: Display Training Results
# ========================================
print("\n📊 Training Results:")
results_dir = Path('runs/detect/isl_model')

# Display training curves
print("\n📈 Training curves:")
if (results_dir / 'results.png').exists():
    display(Image(filename=str(results_dir / 'results.png')))

# Display confusion matrix
print("\n🎯 Confusion Matrix:")
if (results_dir / 'confusion_matrix.png').exists():
    display(Image(filename=str(results_dir / 'confusion_matrix.png')))

# Print metrics
print("\n📊 Final Metrics:")
print(f"  Best model saved at: {results_dir / 'weights' / 'best.pt'}")

# STEP 7: Validate the Model
# ========================================
print("\n🔍 Validating trained model...")
best_model = YOLO(str(results_dir / 'weights' / 'best.pt'))
metrics = best_model.val()

print("\n📊 Validation Metrics:")
print(f"  mAP50: {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall: {metrics.box.mr:.4f}")

# STEP 8: Test Inference
# ========================================
print("\n🎥 Testing inference on validation images...")

# Get a few validation images
val_images_list = list(Path('val/images').glob('*'))[:5]

for img_path in val_images_list:
    print(f"\nProcessing: {img_path.name}")
    results = best_model(str(img_path))
    
    # Save annotated image
    for r in results:
        im_array = r.plot()
        import cv2
        cv2.imwrite(f'prediction_{img_path.name}', im_array)
        display(Image(filename=f'prediction_{img_path.name}'))

# STEP 9: Export for Real-time Inference
# ========================================
print("\n📦 Exporting model for deployment...")

# Export to different formats
print("Exporting to ONNX format...")
best_model.export(format='onnx')

print("\n✅ All done! Your model is ready for real-time ISL detection!")
print(f"\n📁 Model files location:")
print(f"  - PyTorch: {results_dir / 'weights' / 'best.pt'}")
print(f"  - ONNX: {results_dir / 'weights' / 'best.onnx'}")

# STEP 10: Real-time Inference Code Template
# ========================================
print("\n" + "="*60)
print("🎬 REAL-TIME INFERENCE CODE")
print("="*60)
print("""
# Use this code for real-time ISL detection:

from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('runs/detect/isl_model/weights/best.pt')

# For webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inference
    results = model(frame)
    
    # Visualize
    annotated_frame = results[0].plot()
    cv2.imshow('ISL Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
""")