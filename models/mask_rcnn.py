import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from config import NUM_CLASSES, MASK_RCNN_WEIGHTS, DEVICE
class MaskRCNNDetector:
    def __init__(self, num_classes=4, model_path=None, device='cpu'):
        """
        num_classes: số lớp (3 loại tế bào + background)
        model_path: đường dẫn tới file .pth đã fine-tune (nếu có)
        """
        self.device = device
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        mask_in_features = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            mask_in_features, hidden_layer, num_classes
        )
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image, score_thr=0.5):
        """
        image: numpy array (H, W, 3), RGB
        Returns: boxes, scores, labels, masks
        """
        if isinstance(image, np.ndarray):
            image = F.to_tensor(image)
        image = image.to(self.device)
        output = self.model([image])[0]
        keep = output['scores'] > score_thr
        boxes = output['boxes'][keep].cpu().numpy()
        scores = output['scores'][keep].cpu().numpy()
        labels = output['labels'][keep].cpu().numpy()
        masks = output['masks'][keep].cpu().numpy()  # (N, 1, H, W)
        masks = masks[:, 0]  # (N, H, W)
        return boxes, scores, labels, masks