import os
import json
import cv2
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

class FloorplanTransformerDataset(Dataset):
    def __init__(self, json_path, target_size=(512, 512)):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing refined data: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        if args.debug:
            self.data = self.data[:10]
        self.target_size = target_size
        self.model_id = "facebook/mask2former-swin-base-coco-instance"
        self.processor = Mask2FormerImageProcessor.from_pretrained(self.model_id)
        self.class_map = {
            'wall': 1, 'door': 2, 'window': 3, 'kitchen': 4, 
            'livingroom': 5, 'bedroom': 6, 'bathroom': 7
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = cv2.imread(entry['image_path'])
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        semantic_map = np.zeros((h_orig, w_orig), dtype=np.int32)
        for ann in entry['annotations']:
            label = ann['label'].lower()
            if label in self.class_map:
                poly = np.array(ann['polygon'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(semantic_map, [poly], self.class_map[label])
        instance_id_to_semantic_id = {i: i for i in range(len(self.class_map) + 1)}
        return self.processor(
            image, 
            segmentation_maps=semantic_map.astype(np.int64), 
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt"
        )

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"][0] for x in batch]),
        "mask_labels": [x["mask_labels"][0] for x in batch],
        "class_labels": [x["class_labels"][0] for x in batch]
    }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "floorplan_results/mask2former_instance"
    os.makedirs(output_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, 'results.csv')
    history = []
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-instance",
        num_labels=7,
        ignore_mismatched_sizes=True
    ).to(device)
    train_loader = DataLoader(
        FloorplanTransformerDataset('refined_data/train.json'), 
        batch_size=4, 
        shuffle=True, 
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler('cuda')
    epochs = 2 if args.debug else 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            mask_labels = [m.to(device, non_blocking=True) for m in batch["mask_labels"]]
            class_labels = [c.to(device, non_blocking=True) for c in batch["class_labels"]]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                outputs = model(
                    pixel_values=pixel_values, 
                    mask_labels=mask_labels, 
                    class_labels=class_labels
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        history.append({'epoch': epoch, 'train/loss': avg_loss})
        pd.DataFrame(history).to_csv(results_csv, index=False)
        torch.save(model.state_dict(), os.path.join(output_dir, f"m2f_epoch_{epoch}.pt"))

if __name__ == "__main__":
    train()