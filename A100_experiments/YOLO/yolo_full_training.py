import os
import json
import cv2
import shutil
import numpy as np
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
DATA_ROOT = 'yolo_comparison_data'
RESULTS_DIR = 'yolo_thesis_results'
STRUCTURAL_ANCHORS = ['wall', 'door', 'window', 'stairs', 'railing', 'column', 'fixedfurniture', 'fixedfurnitureset', 'opentobelow']

def get_hierarchical_label(raw_label):
    parts = raw_label.strip().lower().split()
    if 'space' in parts:
        idx = parts.index('space')
        if idx + 1 < len(parts):
            if parts[idx + 1] == 'opentobelow': return 'opentobelow'
            return parts[idx + 1]
    if 'fixedfurniture' in parts:
        idx = parts.index('fixedfurniture')
        if idx + 1 < len(parts) and parts[idx + 1] not in ['selectioncontrols', 'dimensionmark']:
            return parts[idx + 1]
        return 'fixedfurniture'
    found_anchor = None
    for part in parts:
        if part in STRUCTURAL_ANCHORS: found_anchor = part
    return found_anchor

def prepare_yolo_data():
    all_found_classes = set()
    files = [('train', 'refined_data/train.json'), ('val', 'refined_data/test.json')]
    
    # First pass: identify classes
    for _, json_path in files:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                for ann in entry.get('annotations', []):
                    clean = get_hierarchical_label(ann.get('label', ''))
                    if clean: all_found_classes.add(clean)
    
    sorted_classes = sorted(list(all_found_classes))
    class_map = {cls: i for i, cls in enumerate(sorted_classes)}
    
    # Second pass: Generate YOLO labels
    for split, json_path in files:
        img_out = os.path.join(DATA_ROOT, 'images', split)
        lbl_out = os.path.join(DATA_ROOT, 'labels', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            img = cv2.imread(entry['image_path'])
            if img is None: continue
            h, w = img.shape[:2]
            file_id = entry['folder_id'].replace('/', '_').replace('\\', '_')
            
            shutil.copy2(entry['image_path'], os.path.join(img_out, f"{file_id}.png"))
            
            with open(os.path.join(lbl_out, f"{file_id}.txt"), 'w') as lf:
                for ann in entry['annotations']:
                    label = get_hierarchical_label(ann['label'])
                    if label in class_map:
                        poly = np.array(ann['polygon'])
                        # Handle normalized vs pixel coordinates
                        if poly.max() > 1.01:
                            poly = poly / [w, h]
                        
                        if len(poly) < 3: continue
                        poly_str = " ".join([f"{c:.6f}" for pair in poly for c in pair])
                        lf.write(f"{class_map[label]} {poly_str}\n")
    
    # Create YAML
    yaml_dict = {
        'path': os.path.abspath(DATA_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: cls for i, cls in enumerate(sorted_classes)}
    }
    with open('cubicasa_yolo.yaml', 'w') as f:
        yaml.dump(yaml_dict, f)
    
    return sorted_classes, class_map

def train_and_evaluate():
    device = 0 if torch.cuda.is_available() else 'cpu'
    sorted_labels, label2id = prepare_yolo_data()
    
    # 1. TRAINING
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data="cubicasa_yolo.yaml",
        epochs=100, # Adjust for your needs
        imgsz=1024, # High res to capture thin architectural lines
        batch=16,
        device=device,
        project=RESULTS_DIR,
        name="comparison_run"
    )

    # 2. EVALUATION (mIoU Calculation)
    # We use the same metric logic as the DINO script for a fair comparison
    best_model = YOLO(f"{RESULTS_DIR}/comparison_run/weights/best.pt")
    
    num_classes = len(sorted_labels) + 1
    total_i = torch.zeros(num_classes).to('cuda' if device == 0 else 'cpu')
    total_u = torch.zeros(num_classes).to('cuda' if device == 0 else 'cpu')

    # Load test data manually to calculate per-pixel mIoU
    with open('refined_data/test.json', 'r') as f:
        test_data = json.load(f)

    print("Calculating mIoU for YOLO Baseline...")
    for entry in tqdm(test_data):
        img = cv2.imread(entry['image_path'])
        if img is None: continue
        h, w = img.shape[:2]
        
        # Inference
        results = best_model.predict(img, imgsz=1024, conf=0.25, verbose=False)[0]
        
        # Create Ground Truth Semantic Map
        gt_map = np.zeros((h, w), dtype=np.int32)
        for ann in entry['annotations']:
            clean = get_hierarchical_label(ann['label'])
            if clean in label2id:
                poly = np.array(ann['polygon'])
                if poly.max() <= 1.01:
                    poly[:, 0] *= w
                    poly[:, 1] *= h
                cv2.fillPoly(gt_map, [poly.astype(np.int32)], label2id[clean] + 1)
        
        gt_torch = torch.from_numpy(gt_map).to(total_i.device)
        
        # Calculate Intersection/Union from YOLO masks
        if results.masks is not None:
            for i, mask_data in enumerate(results.masks.data):
                cls_id = int(results.boxes.cls[i]) + 1
                # Resize mask to original image size
                mask_resized = torch.nn.functional.interpolate(
                    mask_data.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear'
                ).squeeze() > 0.5
                
                total_i[cls_id] += (mask_resized & (gt_torch == cls_id)).sum()
                total_u[cls_id] += (mask_resized | (gt_torch == cls_id)).sum()

    # Final Metric Output
    iou_per_class = total_i[1:] / (total_u[1:] + 1e-6)
    valid_mask = total_u[1:] > 0
    mIoU = iou_per_class[valid_mask].mean().item()
    
    # Save CSV
    results_df = pd.DataFrame({
        "class": sorted_labels,
        "iou": iou_per_class.cpu().numpy(),
        "present": valid_mask.cpu().numpy()
    })
    results_df.to_csv(f"{RESULTS_DIR}/yolo_miou_metrics.csv", index=False)
    
    print(f"\n--- YOLO Comparison Results ---")
    print(f"Overall mIoU: {mIoU:.4f}")
    print(results_df[results_df['present']].sort_values(by='iou', ascending=False).head(10))

if __name__ == "__main__":
    train_and_evaluate()