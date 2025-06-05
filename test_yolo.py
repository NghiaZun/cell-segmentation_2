import cv2
import matplotlib.pyplot as plt
from models.yolo import YOLODetector

# Khởi tạo detector
detector = YOLODetector(model_path='weights/best_yolov8.pt', device='cpu')

# Đọc ảnh
img = cv2.imread('C:/Users/TRUNG NGHIA/Downloads/data/test/img/7ae19de7bc2a.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Dự đoán
detections = detector.predict(img_rgb)

# Vẽ bounding box
for det in detections:
    x1, y1, x2, y2, conf, cls = map(int, det[:6])
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(img_rgb, f'{cls} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

# Hiển thị ảnh
plt.figure(figsize=(10,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()