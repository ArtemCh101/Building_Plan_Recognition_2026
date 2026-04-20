import os
import warnings

warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import cv2
import shutil
import numpy as np
import torch
import yaml
import logging
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logging.getLogger("ultralytics").setLevel(logging.WARNING)

TARGET_CLASSES = ['wall', 'door', 'window', 'kitchen', 'livingroom', 'bedroom', 'bathroom', 'laundry', 'closet', 'toilet', 'sink']
DISPLAY_FILTER = ['wall', 'kitchen', 'livingroom', 'bedroom', 'bathroom', 'laundry']
DATA_ROOT = 'yolo_seg_data'
WEIGHTS_DIR = 'floorplan_results/sahi_baseline/weights'
LAST_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'last.pt')

def clean_label(label_str):
    label_str = label_str.lower()
    if 'dimensionmark' in label_str: return None
    if 'wall' in label_str: return 'wall'
    if 'door' in label_str: return 'door'
    if 'window' in label_str: return 'window'
    if 'closet' in label_str: return 'closet'
    if 'toilet' in label_str: return 'toilet'
    if 'sink' in label_str: return 'sink'
    if 'bathtub' in label_str: return 'bathtub'
    if 'shower' in label_str: return 'shower'
    if 'space' in label_str: return label_str.split()[-1]
    return None

def run_prep():
    if os.path.exists(DATA_ROOT):
        return
    for split, json_name in [('train', 'train.json'), ('val', 'test.json')]:
        json_file = os.path.join('refined_data', json_name)
        images_out = os.path.join(DATA_ROOT, 'images', split)
        labels_out = os.path.join(DATA_ROOT, 'labels', split)
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)
        with open(json_file, 'r') as f:
            data = json.load(f)
        class_map = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
        for entry in data:
            img_path = entry['image_path']
            if not os.path.exists(img_path): continue
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            file_id = entry['folder_id'].replace('/', '_').replace('\\', '_')
            shutil.copy2(img_path, os.path.join(images_out, f"{file_id}.png"))
            with open(os.path.join(labels_out, f"{file_id}.txt"), 'w') as lf:
                for ann in entry['annotations']:
                    label = clean_label(ann['label'])
                    if label in class_map:
                        cls_id = class_map[label]
                        poly = np.array(ann['polygon'])
                        poly_norm = poly / [w, h]
                        if len(poly_norm) < 3: continue
                        poly_str = " ".join([f"{coord:.6f}" for pair in poly_norm for coord in pair])
                        lf.write(f"{cls_id} {poly_str}\n")
    yaml_data = {
        'path': os.path.abspath(DATA_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: cls for i, cls in enumerate(TARGET_CLASSES)}
    }
    with open('cubicasa_seg.yaml', 'w') as f:
        yaml.dump(yaml_data, f)

if __name__ == "__main__":
    run_prep()
    is_resuming = os.path.exists(LAST_CHECKPOINT)
    model_source = LAST_CHECKPOINT if is_resuming else "yolo11n-seg.pt"
    
    model = YOLO(model_source)
    model.train(
        data="cubicasa_seg.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        workers=16,
        device=0,
        project="floorplan_results",
        name="sahi_baseline",
        amp=True,
        verbose=False,
        plots=False,
        resume=is_resuming,
        exist_ok=True
    )
    
    try:
        best_path = 'floorplan_results/sahi_baseline/weights/best.pt'
        if os.path.exists(best_path):
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='ultralytics',
                model_path=best_path,
                device='cuda:0',
                confidence_threshold=0.3
            )
            
            test_img_dir = os.path.join(DATA_ROOT, 'images', 'val')
            test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.png')][:3]
            
            for i, img_path in enumerate(test_images):
                result = get_sliced_prediction(
                    img_path,
                    detection_model,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    verbose=0
                )
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for obj in result.object_prediction_list:
                    if obj.category.name in DISPLAY_FILTER and obj.mask is not None:
                        mask = obj.mask.full_shape_mask
                        color = np.random.randint(0, 255, (3,)).tolist()
                        img[mask > 0] = img[mask > 0] * 0.5 + np.array(color) * 0.5
                cv2.imwrite(f"sahi_out_{i}.png", cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            print("Training finished but best.pt not found for SAHI.")
    except Exception as e:
        print(f"Evaluation failed: {e}")
