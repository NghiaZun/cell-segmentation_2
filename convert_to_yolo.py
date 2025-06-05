import os
import pandas as pd
import numpy as np

# Map class name to class id
CLASS_MAP = {'shsy5y': 0, 'astro': 1, 'cort': 2}

def rle_decode(mask_rle, shape=(520, 704)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def mask_to_bbox(mask):
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return None
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    return x_min, y_min, x_max, y_max

def bbox_to_yolo(bbox, img_shape):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_shape[1]
    y_center = (y_min + y_max) / 2 / img_shape[0]
    width = (x_max - x_min) / img_shape[1]
    height = (y_max - y_min) / img_shape[0]
    return x_center, y_center, width, height

def convert(csv_path, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for image_id, group in df.groupby('id'):
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        lines = []
        for _, row in group.iterrows():
            mask = rle_decode(row['annotation'])
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            x_center, y_center, w, h = bbox_to_yolo(bbox, mask.shape)
            class_id = CLASS_MAP[row['class']]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    convert("/kaggle/input/celsegmentation/train.csv", "/kaggle/working/labels_yolo")