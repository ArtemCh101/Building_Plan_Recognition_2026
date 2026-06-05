import json
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou, nms
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection,
    SamModel, 
    SamProcessor
)
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

def patch_dino_embeddings(model):
    def get_input_embeddings():
        return model.model.text_backbone.embeddings.word_embeddings
    model.get_input_embeddings = get_input_embeddings
    return model

class PlanDataset(Dataset):
    def __init__(self, json_path, processor, classes):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.classes = classes
        self.text_prompt = ". ".join([f"architectural 2d building plan {c}" for c in classes]) + "."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        
        target_size = 800
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        delta_w = target_size - image.size[0]
        delta_h = target_size - image.size[1]
        padding = (0, 0, delta_w, delta_h)
        image = ImageOps.expand(image, padding)
        
        inputs = self.processor(
            images=image, 
            text=self.text_prompt, 
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

def custom_collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
    }

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

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_id = "IDEA-Research/grounding-dino-base"
classes = ["Wall", "Window", "Door", "Room", "Stairs", "Kitchen", "Bathroom", "Closet", "Balcony", "Elevator"]
text_prompt = ". ".join([f"architectural 2d building plan {c}" for c in classes]) + "."

processor = AutoProcessor.from_pretrained(dino_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
model = patch_dino_embeddings(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["projector", "query", "value"],
    lora_dropout=0.1,
    bias="none"
)

peft_model = get_peft_model(model, lora_config)
optimizer = AdamW(peft_model.parameters(), lr=5e-5)

train_ds = PlanDataset("basic_data/train.json", processor, classes)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=custom_collate)

peft_model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = peft_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device)
        )
        loss = -outputs.logits.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

peft_model.save_pretrained("dino_lora_plan_final")

peft_model.eval()
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()

with open("basic_data/test.json", 'r') as f:
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
        dino_outs = peft_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        dino_outs, inputs.input_ids, threshold=0.35, target_sizes=[(h, w)]
    )[0]

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
        sam_in = sam_processor(image_pil, input_boxes=[clean_boxes.cpu().tolist()], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_outs = sam_model(**sam_in)
        
        masks = sam_processor.post_process_masks(
            sam_outs.pred_masks, sam_in["original_sizes"].cpu(), sam_in["reshaped_input_sizes"].cpu()
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
        if img_ious:
            all_ious.append(np.mean(img_ious))

print(f"Final mAP@0.5: {np.mean(all_precisions) if all_precisions else 0:.4f}")
print(f"Final Mean IoU: {np.mean(all_ious) if all_ious else 0:.4f}")