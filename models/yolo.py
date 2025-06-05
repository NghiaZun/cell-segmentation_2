from ultralytics import YOLO
from config import YOLO_WEIGHTS, DEVICE

class YOLODetector:
    def __init__(self, model_path='best_yolov8.pt', device='cpu'):
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, image):
        """
        image: numpy array (H, W, 3), RGB
        Returns: list of detections [x1, y1, x2, y2, confidence, class]
        """
        results = self.model.predict(source=image, device=self.device, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                detections.append([x1, y1, x2, y2, conf, cls])
        return detections