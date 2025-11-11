from ultralytics import YOLO
import os
from tqdm import tqdm

model = YOLO("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt")

input_folder = r"D:\project\train\images"
output_folder = r"D:\project\train\labels"
os.makedirs(output_folder, exist_ok=True)
           
for cls_folder in os.listdir(input_folder):
    img_dir = os.path.join(input_folder, cls_folder)
    label_dir = os.path.join(output_folder, cls_folder)
    os.makedirs(label_dir, exist_ok=True)

    for file in tqdm(os.listdir(img_dir), desc=f"Labeling {cls_folder}"):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(img_dir, file)
        results = model(img_path, save=False, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            with open(os.path.join(label_dir, file.rsplit('.', 1)[0] + ".txt"), "w") as f:
                for box in boxes:
                    x, y, w, h = box.xywhn[0]
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
