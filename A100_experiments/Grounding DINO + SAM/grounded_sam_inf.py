import os
import numpy as np
import cv2
import json
import pandas as pd
from tqdm import tqdm

STRUCTURAL_ANCHORS = [
    'wall', 'door', 'window', 'stairs', 'railing', 
    'column', 'fixedfurniture', 'fixedfurnitureset', 'opentobelow'
]

def get_hierarchical_label(raw_label):
    parts = raw_label.strip().lower().split()
    if 'space' in parts:
        idx = parts.index('space')
        if idx + 1 < len(parts):
            if parts[idx + 1] == 'opentobelow': return 'opentobelow'
            return parts[idx + 1]
    if 'fixedfurniture' in parts:
        idx = parts.index('fixedfurniture')
        if idx + 1 < len(parts):
            if parts[idx + 1] not in ['selectioncontrols', 'dimensionmark']:
                return parts[idx + 1]
        return 'fixedfurniture'
    if 'fixedfurnitureset' in parts:
        idx = parts.index('fixedfurnitureset')
        if idx + 1 < len(parts): return parts[idx + 1]
        return 'fixedfurniture'
    found_anchor = None
    for part in parts:
        if part in STRUCTURAL_ANCHORS: found_anchor = part
    return found_anchor

def calculate_metrics(results_dir, gt_json_path):
    if not os.path.exists(gt_json_path):
        print(f"Error: Could not find GT file at {gt_json_path}")
        return

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    found_labels = set()
    for entry in gt_data:
        for ann in entry.get('annotations', []):
            clean = get_hierarchical_label(ann.get('label', ''))
            if clean: found_labels.add(clean)
    
    sorted_labels = sorted(list(found_labels))
    label2id = {label: i + 1 for i, label in enumerate(sorted_labels)}
    num_classes = len(sorted_labels) + 1
    
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    print(f"Evaluating {len(gt_data)} images across {len(sorted_labels)} structural classes...")
    
    found_any_results = False
    for idx, entry in enumerate(tqdm(gt_data)):
        image_id = entry.get('id', idx)
        res_file = os.path.join(results_dir, f"res_{image_id}.npz")
        
        if not os.path.exists(res_file):
            continue
        
        found_any_results = True
        
        preds = np.load(res_file, allow_pickle=True)
        pred_masks = preds['masks']   
        pred_labels = preds['labels']
        pred_scores = preds['scores'] if 'scores' in preds else np.ones(len(pred_labels))
        
        h, w = pred_masks.shape[2], pred_masks.shape[3]
        
        pred_map = np.zeros((h, w), dtype=np.int32)
        gt_map = np.zeros((h, w), dtype=np.int32)
        
        for ann in entry.get('annotations', []):
            label = get_hierarchical_label(ann.get('label', ''))
            if label in label2id:
                poly = np.array(ann['polygon'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(gt_map, [poly], label2id[label])
        
        order = np.argsort(pred_scores)
        for i in order:
            clean_label = pred_labels[i]
            if clean_label in label2id:
                mask_binary = pred_masks[i][0] > 0
                pred_map[mask_binary] = label2id[clean_label]

        for cls_id in range(1, num_classes):
            inter = ((pred_map == cls_id) & (gt_map == cls_id)).sum()
            union = ((pred_map == cls_id) | (gt_map == cls_id)).sum()
            total_intersection[cls_id] += inter
            total_union[cls_id] += union

    if not found_any_results:
        print(f"Error: No .npz files found in {results_dir}. Check paths/IDs.")
        return

    valid_classes_mask = total_union[1:] > 0
    iou_per_class = total_intersection[1:] / (total_union[1:] + 1e-6)
    
    miou = np.mean(iou_per_class[valid_classes_mask]) if any(valid_classes_mask) else 0.0
    
    metrics_df = pd.DataFrame({
        'class': sorted_labels,
        'iou': iou_per_class,
        'present_in_gt': valid_classes_mask
    })
    
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "miou_results.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    print(f"\n--- Evaluation Summary ---")
    print(f"Overall mIoU: {miou:.4f}")
    print(f"Results saved to: {csv_path}")
    print("\nTop 10 Classes by IoU:")
    print(metrics_df[metrics_df['present_in_gt']].sort_values(by='iou', ascending=False).head(10))

if __name__ == "__main__":
    calculate_metrics(
        results_dir="floorplan_results/grounded_sam_batched", 
        gt_json_path="refined_data/train.json"
    )