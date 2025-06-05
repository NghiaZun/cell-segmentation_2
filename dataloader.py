import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import IMAGES_DIR, CSV_PATH

class SartoriusDataset(Dataset):
    def __init__(self, transforms=None):
        self.images_dir = IMAGES_DIR
        self.df = pd.read_csv(CSV_PATH)
        self.image_ids = self.df['id'].unique()
        self.transforms = transforms

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

        masks = []
        labels = []
        records = self.df[self.df['id'] == image_id]
        for _, row in records.iterrows():
            mask = self.rle_decode(row['annotation'])
            masks.append(mask)
            labels.append(row['class'])  # class: 'shsy5y', 'astro', 'cort'

        masks = np.stack(masks, axis=0) if masks else np.zeros((0, 520, 704), dtype=np.uint8)

        if self.transforms:
            # Apply transforms (e.g., albumentations)
            augmented = self.transforms(image=image, masks=list(masks))
            image = augmented['image']
            masks = np.stack(augmented['masks'])

        return {
            'image': image,
            'masks': masks,
            'labels': labels,
            'image_id': image_id
        }