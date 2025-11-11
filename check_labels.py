import cv2
import os

# 🔧 Change these paths
image_folder = r"D:\project\A"        # your original images
label_folder = r"D:\project"  # folder where .txt labels are saved

# Loop through a few images
for img_name in os.listdir(image_folder)[:10]:  # check first 10 images
    if not img_name.lower().endswith(('.jpg', '.png')):
        continue

    image_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, img_name.rsplit('.', 1)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    with open(label_path, 'r') as f:
        for line in f.readlines():
            c, x, y, width, height = map(float, line.split())
            x1 = int((x - width / 2) * w)
            y1 = int((y - height / 2) * h)
            x2 = int((x + width / 2) * w)
            y2 = int((y + height / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "A", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Check Label", img)
    key = cv2.waitKey(0)
    if key == 27:  # ESC to exit early
        break

cv2.destroyAllWindows()
