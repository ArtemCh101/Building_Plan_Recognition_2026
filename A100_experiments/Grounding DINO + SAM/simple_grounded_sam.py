import json
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.ops import box_iou, nms
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection,
    SamModel, 
    SamProcessor
)

def calculate_iou_mask(pred_mask, gt_polygon, shape):
    gt_mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(gt_polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(gt_mask, [points], 1)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

def get_gt_boxes(annotations):
    boxes = []
    for ann in annotations:
        poly = ann.get("polygon", [])
        if not poly: continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return torch.tensor(boxes) if boxes else torch.empty((0, 4))

def compute_ap_50(all_results):
    precisions = []
    for res in all_results:
        dt_boxes = res['dt_boxes']
        gt_boxes = res['gt_boxes']
        if len(gt_boxes) == 0: continue
        if len(dt_boxes) == 0:
            precisions.append(0.0)
            continue
            
        ious = box_iou(dt_boxes, gt_boxes)
        matches = (ious >= 0.5).any(dim=1).float()
        precisions.append(matches.mean().item())
    return np.mean(precisions) if precisions else 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"

dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device).eval()

sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()

classes = ["Wall", "Window", "Door", "Room", "Stairs", "Kitchen", "Bathroom", "Closet", "Balcony", "Elevator"]
text_prompt = ". ".join([f"architectural 2d building plan {c}" for c in classes]) + "."

with open("basic_data/test.json", 'r') as f:
    data = json.load(f)

all_ious = []
map_data = []

for img_info in data:
    try:
        image_pil = Image.open(img_info["image_path"]).convert("RGB")
        w, h = image_pil.size
    except: continue

    dino_inputs = dino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        dino_outputs = dino_model(**dino_inputs)

    results = dino_processor.post_process_grounded_object_detection(
        dino_outputs, dino_inputs.input_ids, threshold=0.30, target_sizes=[(h, w)]
    )[0]

    keep = nms(results["boxes"], results["scores"], iou_threshold=0.5)
    clean_boxes = results["boxes"][keep]
    clean_scores = results["scores"][keep]

    gt_boxes = get_gt_boxes(img_info.get("annotations", []))
    map_data.append({'dt_boxes': clean_boxes.cpu(), 'gt_boxes': gt_boxes})

    if len(clean_boxes) > 0:
        high_conf = clean_scores > 0.4
        sam_boxes = clean_boxes[high_conf].cpu().tolist()
        
        if not sam_boxes: continue

        sam_inputs = sam_processor(image_pil, input_boxes=[sam_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)

        masks = sam_processor.post_process_masks(
            sam_outputs.pred_masks, sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
        )[0]
        
        binary_masks = (masks[:, 0, :, :] > 0.0).cpu().numpy()
        
        img_ious = []
        for pred_m in binary_masks:
            max_iou = 0
            for ann in img_info.get("annotations", []):
                poly = ann.get("polygon", [])
                if not poly: continue
                iou = calculate_iou_mask(pred_m, poly, (h, w))
                max_iou = max(max_iou, iou)
            img_ious.append(max_iou)
        all_ious.append(np.mean(img_ious))

print(f"Mean IoU (Segmentation): {np.mean(all_ious):.4f}")
print(f"mAP@0.5 (Detection): {compute_ap_50(map_data):.4f}")