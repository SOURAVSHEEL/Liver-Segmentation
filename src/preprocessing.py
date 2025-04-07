import os 
import cv2
import json
import logging
import numpy as np
from tqdm import tqdm
from src.logs import *  

# Get loggers
train_logger, error_logger = get_loggers()

LABELS = {"liver":1,
          "mass":2,
          "outline":3}

def process_image(image_path,seg_base_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image is None or cannot be read.")
        
        mask = np.zeros(img.shape[:2],dtype=np.uint8)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        train_logger.info(f"Processing images: {img_name}")


        for region, label in LABELS.items():
            region_path = os.path.join(seg_base_path, region, img_name + ".json")
            if os.path.exists(region_path):
                with open(region_path, 'r') as f:
                    data = json.load(f)
                    if  region in data:
                        points = np.array(data[region],dtype=np.int32)
                        if points.ndim == 2:
                            cv2.fillPoly(mask, [points], label)
                        else:
                            error_logger.error(f"Invalid shape in {region_path}")
                    else:
                        error_logger.error(f"key '{region}' missing in {region_path}")
            else:
                error_logger.error(f"Missing file: {region_path}")

        return  mask
    
    except Exception as e:
        error_logger.error(f"Error processing {image_path}: {e}")
        return None
    

def convert_all_categories(root_dir, save_root):
    os.makedirs(save_root,exist_ok=True)
    for category_folder in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_folder)
        if not os.path.isdir(category_path):
            continue

        image_dir = os.path.join(category_path, "image")
        seg_dir = os.path.join(category_path, "segmentation")
        save_dir = os.path.join(save_root, category_folder.replace("\\","_"))
        os.makedirs(save_dir,exist_ok=True)

        for fname in tqdm(os.listdir(image_dir), desc=f"Processing {category_folder}"):
            if not fname.endswith((".png",".jpg","jpeg")):
                continue
            image_path = os.path.join(image_dir,fname)
            mask =  process_image(image_path,seg_dir)

            if mask is not None:
                save_path = os.path.join(save_dir,fname.replace(".jpg",".png").replace(".jpeg",".png"))
                cv2.imwrite(save_path,mask)
                train_logger.info(f"Saved mask: {save_path}")


convert_all_categories("data/raw","data/masks")

