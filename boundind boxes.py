import cv2
import os

image_folder = r"D:\project\train\images"
label_folder = r"D:\project\train\labels"

for img_file in os.listdir(image_folder):
    if not img_file.endswith(".jpg"):
        continue

    img_path = os.path.join(image_folder, img_file)
    label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        for line in f:
            _, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Labeled Image", img)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
