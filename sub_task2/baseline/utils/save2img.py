import torch
import random
import numpy as np
import cv2
import os

def save2img(imgs:torch.tensor, epoch:int, sampling:float=0.3, save_path:str='save'):
    
    batch_size = imgs.size(0)
    indexing = list(range(0, batch_size))
    select_idx = random.sample(indexing, int(batch_size * sampling))

    save_path = os.path.join(save_path, 'img')
    os.makedirs(save_path, exist_ok=True)

    for idx in select_idx:
        img_np = imgs[idx].numpy().transpose(1, 2, 0).copy() 
        img_np = (img_np * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_path, f"augmented_{epoch}_{idx}.png"), img_np)

