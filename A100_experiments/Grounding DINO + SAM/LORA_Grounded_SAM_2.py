import json
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.ops import box_iou, nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor

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
        if not poly:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return torch.tensor(boxes) if boxes else torch.empty((0, 4))

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_id = "IDEA-Research/grounding-dino-base"
classes = ["Wall", "Window", "Door", "Room", "Stairs", "Kitchen", "Bathroom", "Closet", "Balcony", "Elevator"]
text_prompt = ". ".join([f"architectural 2d building plan {c}" for c in classes]) + "."

processor = AutoProcessor.from_pretrained(dino_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()

sam_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
sam_model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(device).eval()

with open("basic_data/test.json", "r") as f:
    test_data = json.load(f)

all_ious = []
all_precisions = []

for img_info in test_data:
    try:
        image_pil = Image.open(img_info["image_path"]).convert("RGB")
        w, h = image_pil.size
    except:
        continue

    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        dino_outs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        dino_outs, inputs.input_ids, threshold=0.35, target_sizes=[(h, w)]
    )[0]

    if len(results["boxes"]) == 0:
        all_precisions.append(0.0)
        continue

    keep = nms(results["boxes"], results["scores"], iou_threshold=0.5)
    clean_boxes = results["boxes"][keep]
    
    gt_boxes = get_gt_boxes(img_info.get("annotations", []))
    if len(gt_boxes) > 0:
        if len(clean_boxes) > 0:
            ious = box_iou(clean_boxes.cpu(), gt_boxes)
            matches = (ious >= 0.5).any(dim=1).float()
            all_precisions.append(matches.mean().item())
        else:
            all_precisions.append(0.0)

    if len(clean_boxes) > 0:
        sam_in = sam_processor(images=image_pil, input_boxes=[clean_boxes.cpu().tolist()], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_outs = sam_model(**sam_in)
        
        masks = sam_processor.post_process_masks(
            sam_outs.pred_masks, sam_in["original_sizes"]
        )[0]
        
        binary_masks = (masks[:, 0, :, :] > 0.0).cpu().numpy()
        
        img_ious = []
        for pred_m in binary_masks:
            max_iou = 0
            for ann in img_info.get("annotations", []):
                poly = ann.get("polygon", [])
                if not poly:
                    continue
                iou = calculate_iou_mask(pred_m, poly, (h, w))
                max_iou = max(max_iou, iou)
            img_ious.append(max_iou)
        if img_ious:
            all_ious.append(np.mean(img_ious))

print(f"Final mAP@0.5: {np.mean(all_precisions) if all_precisions else 0:.4f}")
print(f"Final Mean IoU: {np.mean(all_ious) if all_ious else 0:.4f}")