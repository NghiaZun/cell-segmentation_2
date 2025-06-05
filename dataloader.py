import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SartoriusDataset(Dataset):
    def __init__(self, images_dir, csv_path, label_path, image_ids=None, transforms=None):
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = self.df['id'].unique()
        self.transforms = transforms
        self.label_path = label_path

    def __len__(self):
        return len(self.image_ids)

    def rle_decode(self, mask_rle, shape=(520, 704)):
        '''
        Decode run-length encoded mask.
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_id}.png")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        if os.path.exists(self.label_path):
            label_path = os.path.join(self.label_path, f"{image_id}.txt")
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    # Chuyển từ YOLO format về [xmin, ymin, xmax, ymax] (nếu cần)
                    img_h, img_w = image.shape[:2]
                    xc = x_center * img_w
                    yc = y_center * img_h
                    bw = w * img_w
                    bh = h * img_h
                    xmin = xc - bw / 2
                    ymin = yc - bh / 2
                    xmax = xc + bw / 2
                    ymax = yc + bh / 2
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transforms:
            # Apply transforms (nếu có)
            augmented = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            image = augmented['image']
            boxes = np.array(augmented['bboxes'])
            labels = np.array(augmented['class_labels'])

        return image, boxes, labels, image_id