import json
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.ops import box_iou, nms
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection,
    Sam2Model, 
    Sam2Processor
)

def calculate_iou_mask(pred_mask, gt_polygon, shape):
    gt_mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(gt_polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(gt_mask, [points], 1)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

def get_gt_data(annotations):
    data = {}
    for ann in annotations:
        cls = ann.get("class")
        poly = ann.get("polygon", [])
        if not poly: 
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        box = torch.tensor([min(xs), min(ys), max(xs), max(ys)])
        if cls not in data:
            data[cls] = {"boxes": [], "polygons": []}
        data[cls]["boxes"].append(box)
        data[cls]["polygons"].append(poly)
    
    for cls in data:
        data[cls]["boxes"] = torch.stack(data[cls]["boxes"])
    return data

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_id = "IDEA-Research/grounding-dino-base"
sam2_id = "facebook/sam2.1-hiera-large"

processor = AutoProcessor.from_pretrained(dino_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()
sam2_processor = Sam2Processor.from_pretrained(sam2_id)
sam2_model = Sam2Model.from_pretrained(sam2_id).to(device).eval()

classes = ["Wall", "Window", "Door", "Room", "Stairs", "Kitchen", "Bathroom", "Closet", "Balcony", "Elevator"]

prompts = {
    "Synonyms": {
        "text": "structural walls or partitions. windows, glass panes, or glazing. doors, entrances, or doorways. rooms, indoor spaces, or chambers. stairs, staircases, or steps. kitchen cooking areas or kitchenettes. bathrooms, restrooms, or lavatories. closets or storage spaces. balconies, terraces, or verandas. elevators or lift shafts.",
        "mappings": {
            "Wall": ["wall", "partition"],
            "Window": ["window", "glass", "glazing"],
            "Door": ["door", "entrance", "doorway"],
            "Room": ["room", "space", "chamber"],
            "Stairs": ["stair", "step"],
            "Kitchen": ["kitchen"],
            "Bathroom": ["bathroom", "restroom", "lavatory"],
            "Closet": ["closet", "storage"],
            "Balcony": ["balcony", "terrace", "veranda"],
            "Elevator": ["elevator", "lift"]
        }
    }
}

with open("basic_data/test.json", 'r') as f:
    test_data = json.load(f)

for p_name, p_config in prompts.items():
    print(f"Running Evaluation for Prompt Style: {p_name}")
    all_ious, all_precisions = [], []
    p_text = p_config["text"]
    mappings = p_config["mappings"]

    for img_info in test_data:
        try:
            image_pil = Image.open(img_info["image_path"]).convert("RGB")
            w, h = image_pil.size
        except: 
            continue

        inputs = processor(images=image_pil, text=p_text, return_tensors="pt").to(device)
        with torch.no_grad():
            dino_outs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            dino_outs, inputs.input_ids, threshold=0.30, target_sizes=[(h, w)]
        )[0]

        gt_data = get_gt_data(img_info.get("annotations", []))
        
        for target_cls in classes:
            keywords = mappings[target_cls]
            indices = [
                i for i, label in enumerate(results["labels"]) 
                if any(kw in label.lower() for kw in keywords)
            ]
            
            if not indices:
                if target_cls in gt_data:
                    all_precisions.append(0.0)
                continue
            
            cls_boxes = results["boxes"][indices]
            cls_scores = results["scores"][indices]
            keep = nms(cls_boxes, cls_scores, iou_threshold=0.5)
            final_boxes = cls_boxes[keep]

            if target_cls in gt_data:
                ious = box_iou(final_boxes.cpu(), gt_data[target_cls]["boxes"])
                all_precisions.append((ious >= 0.5).any(dim=1).float().mean().item())

                sam_in = sam2_processor(images=image_pil, input_boxes=[final_boxes.cpu().tolist()], return_tensors="pt").to(device)
                with torch.no_grad():
                    sam_outs = sam2_model(**sam_in)
                
                masks = sam2_processor.post_process_masks(
                    sam_outs.pred_masks, sam_in["original_sizes"]
                )[0]
                
                binary_masks = (masks[:, 0, :, :] > 0.0).cpu().numpy()
                for pred_m in binary_masks:
                    max_iou = max([calculate_iou_mask(pred_m, poly, (h, w)) for poly in gt_data[target_cls]["polygons"]])
                    all_ious.append(max_iou)

    print(f"Results for {p_name}: mAP@0.5: {np.mean(all_precisions):.4f}, mIoU: {np.mean(all_ious):.4f}\n")