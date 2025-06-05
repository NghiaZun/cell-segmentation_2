import torch
from torch.utils.data import DataLoader
from models.mask_rcnn import MaskRCNNDetector
from dataloader import SartoriusDataset
from config import TRAIN_IMAGES_DIR,VAL_IMAGES_DIR, CSV_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, NUM_CLASSES, CLASS_NAMES, MASK_RCNN_WEIGHTS
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def train_mask_rcnn():
    train_image_ids = [os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMAGES_DIR)]
    val_image_ids = [os.path.splitext(f)[0] for f in os.listdir(VAL_IMAGES_DIR)]
    train_dataset = SartoriusDataset(TRAIN_IMAGES_DIR, CSV_PATH, image_ids=train_image_ids, transforms=None)
    val_dataset = SartoriusDataset(VAL_IMAGES_DIR, CSV_PATH, image_ids=val_image_ids, transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)   
    model = MaskRCNNDetector(num_classes=NUM_CLASSES, model_path=None, device=DEVICE).model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # Train loop
        model.train()
        for images, masks, labels, image_ids in train_loader:
            images = [torch.as_tensor(img, dtype=torch.float32).permute(2,0,1)/255. for img in images]
            targets = []
            for i in range(len(images)):
                # ...tạo bboxes từ masks ở đây...
                target = {
                    "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
                    "labels": torch.as_tensor([CLASS_NAMES[l] for l in labels[i]], dtype=torch.int64),
                    "masks": torch.as_tensor(masks[i], dtype=torch.uint8)
                }
                targets.append(target)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {losses.item()}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, labels, image_ids in val_loader:
                images = [torch.as_tensor(img, dtype=torch.float32).permute(2,0,1)/255. for img in images]
                targets = []
                for i in range(len(images)):
                    # ...tạo bboxes từ masks ở đây...
                    target = {
                        "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
                        "labels": torch.as_tensor([CLASS_NAMES[l] for l in labels[i]], dtype=torch.int64),
                        "masks": torch.as_tensor(masks[i], dtype=torch.uint8)
                    }
                    targets.append(target)
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), MASK_RCNN_WEIGHTS)

if __name__ == "__main__":
    train_mask_rcnn()