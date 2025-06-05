import torch
from torch.utils.data import DataLoader
from models.mask_rcnn import MaskRCNNDetector
from dataloader import SartoriusDataset
from config import IMAGES_DIR, CSV_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, NUM_CLASSES, CLASS_NAMES, MASK_RCNN_WEIGHTS

def collate_fn(batch):
    return tuple(zip(*batch))

def train_mask_rcnn():
    dataset = SartoriusDataset(IMAGES_DIR, CSV_PATH, transforms=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = MaskRCNNDetector(num_classes=NUM_CLASSES, model_path=None, device=DEVICE).model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for images, masks, labels, image_ids in dataloader:
            images = [torch.as_tensor(img, dtype=torch.float32).permute(2,0,1)/255. for img in images]
            targets = []
            for i in range(len(images)):
                target = {
                    "boxes": torch.as_tensor(bboxes, dtype=torch.float32),   # Tạo bounding box từ mask
                    "labels": torch.as_tensor([CLASS_NAMES[l] for l in labels[i]], dtype=torch.int64),  # Chuyển label sang số
                    "masks": torch.as_tensor(masks[i], dtype=torch.uint8)
                }
                targets.append(target)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {losses.item()}")
        torch.save(model.state_dict(), MASK_RCNN_WEIGHTS)

if __name__ == "__main__":
    train_mask_rcnn()