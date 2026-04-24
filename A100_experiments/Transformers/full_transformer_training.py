import os
import json
import cv2
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

STRUCTURAL_ANCHORS = [
    'wall', 'door', 'window', 'stairs', 'railing', 
    'column', 'fixedfurniture', 'fixedfurnitureset', 'opentobelow'
]

def get_hierarchical_label(raw_label):
    parts = raw_label.strip().lower().split()
    if 'space' in parts:
        idx = parts.index('space')
        if idx + 1 < len(parts):
            if parts[idx + 1] == 'opentobelow':
                return 'opentobelow'
            return parts[idx + 1]
    if 'fixedfurniture' in parts:
        idx = parts.index('fixedfurniture')
        if idx + 1 < len(parts):
            if parts[idx + 1] not in ['selectioncontrols', 'dimensionmark']:
                return parts[idx + 1]
        return 'fixedfurniture'
    if 'fixedfurnitureset' in parts:
        idx = parts.index('fixedfurnitureset')
        if idx + 1 < len(parts):
            return parts[idx + 1]
        return 'fixedfurniture'
    found_anchor = None
    for part in parts:
        if part in STRUCTURAL_ANCHORS:
            found_anchor = part
    return found_anchor

class FloorplanTransformerDataset(Dataset):
    def __init__(self, json_path, processor=None, class_map=None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing data: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        if args.debug:
            self.data = self.data[:10]
        
        self.processor = processor if processor else Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance")
        
        if class_map:
            self.class_map = class_map
        else:
            found_labels = set()
            for entry in self.data:
                for ann in entry.get('annotations', []):
                    clean = get_hierarchical_label(ann.get('label', ''))
                    if clean: found_labels.add(clean)
            self.sorted_labels = sorted(list(found_labels))
            self.class_map = {label: i + 1 for i, label in enumerate(self.sorted_labels)}
        
        self.num_classes = len(self.class_map)

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
        annotations = sorted(entry.get('annotations', []), key=lambda x: 'wall' in x.get('label', '').lower())
        
        for ann in annotations:
            clean = get_hierarchical_label(ann.get('label', ''))
            if clean in self.class_map:
                poly = np.array(ann['polygon'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(semantic_map, [poly], self.class_map[clean])
        
        inputs = self.processor(image, segmentation_maps=semantic_map.astype(np.int64), instance_id_to_semantic_id={i: i for i in range(self.num_classes + 1)}, return_tensors="pt")
        
        inputs = {k: v[0] for k, v in inputs.items()}
        
        semantic_map_res = cv2.resize(semantic_map, (512, 512), interpolation=cv2.INTER_NEAREST)
        inputs['gt_semantic_map'] = torch.from_numpy(semantic_map_res).long()
        
        return inputs

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "mask_labels": [x["mask_labels"] for x in batch],
        "class_labels": [x["class_labels"] for x in batch],
        "gt_semantic_map": torch.stack([x["gt_semantic_map"] for x in batch])
    }

def calculate_iou(pred_map, gt_map, num_classes):
    iou_list = []
    for cls in range(1, num_classes + 1):
        intersection = ((pred_map == cls) & (gt_map == cls)).sum().item()
        union = ((pred_map == cls) | (gt_map == cls)).sum().item()
        if union > 0:
            iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0.0

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "floorplan_results/mask2former_full"
    os.makedirs(output_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, 'results.csv')
    
    train_dataset = FloorplanTransformerDataset('refined_data/train.json')
    val_dataset = FloorplanTransformerDataset('refined_data/test.json', processor=train_dataset.processor, class_map=train_dataset.class_map)
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-instance",
        num_labels=train_dataset.num_classes,
        ignore_mismatched_sizes=True
    ).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler('cuda')
    history = []
    
    epochs = 2 if args.debug else 150
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        total_iou = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                gt_maps = batch["gt_semantic_map"].to(device)
                
                outputs = model(pixel_values=pixel_values)
                mask_preds = train_dataset.processor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512)] * len(pixel_values))
                
                for i in range(len(mask_preds)):
                    total_iou += calculate_iou(mask_preds[i], gt_maps[i], train_dataset.num_classes)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_iou = total_iou / len(val_dataset)
        
        print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | mIoU: {avg_val_iou:.4f}")
        history.append({'epoch': epoch, 'train/loss': avg_train_loss, 'metrics/mIoU': avg_val_iou})
        pd.DataFrame(history).to_csv(results_csv, index=False)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_map': train_dataset.class_map,
            'num_labels': train_dataset.num_classes
        }, os.path.join(output_dir, f"m2f_epoch_{epoch}.pt"))

if __name__ == "__main__":
    train()