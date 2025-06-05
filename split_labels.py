def convert(csv_path, label_dir, image_ids=None):
    os.makedirs(label_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if image_ids is not None:
        df = df[df['id'].isin(image_ids)]
    for image_id, group in df.groupby('id'):
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        lines = []
        for _, row in group.iterrows():
            mask = rle_decode(row['annotation'])
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            x_center, y_center, w, h = bbox_to_yolo(bbox, mask.shape)
            class_id = CLASS_MAP[row['cell_type']]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

# Ví dụ dùng cho train:
import glob

train_img_dir = "/kaggle/input/celsegmentation/archive/data/train/img"
train_image_ids = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{train_img_dir}/*.png")]
convert("/kaggle/input/celsegmentation/train.csv", "/kaggle/working/train/labels", train_image_ids)

# Tương tự cho val:
val_img_dir = "/kaggle/input/celsegmentation/archive/data/val/img"
val_image_ids = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{val_img_dir}/*.png")]
convert("/kaggle/input/celsegmentation/train.csv", "/kaggle/working/val/labels", val_image_ids)