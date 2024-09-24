#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      : pytorch_inference.py
@Time      : 2024/09/13 20:01:27
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''


'''
source code borrow from: https://huggingface.co/spaces/briaai/BRIA-RMBG-1.4/blob/main/app.py
'''

import os
import cv2
import time
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from briarmbg import BriaRMBG


class BriaRMBGPipe(nn.Module):
    def __init__(self, model_path: str="briaai/RMBG-1.4", gpu_id: int=0):
        super().__init__()
        self.model = BriaRMBG.from_pretrained(model_path)
        self.device = torch.device(f"cuda:{str(gpu_id)}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.normalize_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.normalize_std = torch.tensor([1.0, 1.0, 1.0], device=self.device).view(1, 3, 1, 1)
        
    def preprocess(self, img_rgb: Tensor):
        '''
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        '''
        img_tensor = img_tensor.to(self.device).float()
        img_resized = F.interpolate(img_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
        img_resized = img_resized / 255.0
        img_normalized = (img_resized - self.normalize_mean) / self.normalize_std

        return img_normalized

    @torch.no_grad()
    def forward(self, img_rgb: Tensor):
        size = img_rgb.shape[2:]
        img_tensor = self.preprocess(img_rgb)
        preds = self.model(img_tensor)
        preds = F.interpolate(preds[0][0], size=tuple(size), mode='bilinear')
        ma = torch.max(preds)
        mi = torch.min(preds)
        mask = (preds-mi)/(ma-mi)
        # mask = (mask*255).to(torch.uint8)

        return mask


if __name__ == "__main__":
    rmbg = BriaRMBGPipe("RMBG-1.4") # hugging face
    
    save_root = "result"
    os.makedirs(save_root, exist_ok=True)
    img_paths = glob(os.path.join("data", "*.jpg"))

    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split(".")[0]
        img = cv2.imread(img_path)
        img_tensor = torch.from_numpy(img).unsqueeze(0)[..., (2, 1, 0)].permute(0, 3, 1, 2)  # bgr -> rgb
        # print(img_tensor.shape)
        size = tuple(img_tensor.shape[2:])
        
        # inference
        t0 = time.time()
        preds = rmbg(img_tensor)
        print(f"inference use time: {(time.time()-t0):.3f} s")
        
        # print(preds.shape)
        mask = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # print(mask.shape)
        
        # save as 4 channel image
        result = np.zeros((size[0], size[1]*2, 4))
        result[:, :size[1], :3] = img
        result[:, :size[1], 3] = np.ones(size) * 255
        result[:, size[1]:, :3] = img
        result[:, size[1]:, 3] = mask[:, :, 0] * 255.0
        cv2.imwrite(os.path.join(save_root, f"{img_name}_matting.png"), result)

        # save as 3 channel image
        result = img * mask
        cv2.imwrite(os.path.join(save_root, f"{img_name}_result.png"), result)
