import os
import os.path as osp
import cv2
import numpy as np
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
from loguru import logger


LOGDIR = "logs"
if not osp.exists(LOGDIR):
    os.makedirs(LOGDIR)
logger.add(osp.join(LOGDIR, f"merge_{timestr}.log"))
    

def weighted_fusion(img1, img2):
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    w1 = 0.6
    w2 = 1-w1
    logger.info(f'w1={w1}, w2={w2}')
    ensemble = (w1* img1_float + w2 * img2_float).clip(0, 255).astype(np.uint8)
    

    return ensemble


def process_folders(folder1, folder2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames1 = sorted(os.listdir(folder1))
   
    for filename in filenames1:
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
       
        
        if not os.path.exists(path2):  
            logger.info(f"Skip {filename}, there is no corresponding file in {folder2}")
            continue

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
       

        if img1 is None or img2 is None:
            logger.info(f"skip {filename}, cannot read image")
            continue

        
        if img1.shape != img2.shape:
            logger.info(f"skip {filename}, image size not match: [img1]{img1.shape} vs [img2]{img2.shape}")
            continue

        fused_img = weighted_fusion(img1, img2)

        
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, fused_img)

        logger.info(f"processed: {filename}")


folder1 = "EFNet_output"  
folder2 = "STCNet_output"  

output_folder = "BUPTMM_fused"

process_folders(folder1, folder2, output_folder)