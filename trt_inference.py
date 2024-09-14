#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      : trt_inference.py
@Time      : 2024/09/12 15:24:42
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
scrpit input and output are same as BiRefNet_TRT
'''

import os
import cv2
import time
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from tensorrt_base_v4 import TensorrtBaseV4   # same as BiRefNet/script/tensorrt_base_v4.py


class BriaRMBGMatting(TensorrtBaseV4):
    def __init__(self, engine_file_path='', gpu_id=0):
        profiles_max_shapes = [{
            0: (1, 3, 1024, 1024),  # RGB
            1: (1, 1, 1024, 1024),
        }]
        super(BriaRMBGMatting, self).__init__(plan_file_path=engine_file_path, profiles_max_shapes=profiles_max_shapes, gpu_id=gpu_id)
    
    def __call__(self, images):
        profile_num = 0
        img_h, img_w = images.shape[2:]
        bufferH = [
            images.to(torch.uint8, copy=False).contiguous(),
            torch.empty((1, 1, img_h, img_w), dtype=torch.float32, device=f'cuda:{self.gpu_id}'),
        ]
        outputs = self.do_inference(bufferH, profile_num)
        return outputs[0]

    def preheat(self):
        images = torch.zeros((1, 3, 512, 512))
        self.__call__(images)

if __name__ == "__main__":
    trt_rmbg = BriaRMBGMatting("RMBG_1.4.plan")

    save_root = "result_trt"
    os.makedirs(save_root, exist_ok=True)
    img_paths = glob(os.path.join("data", "*.jpg"))

    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split(".")[0]
        img = cv2.imread(img_path)
        img_tensor = torch.from_numpy(img).unsqueeze(0)[..., (2, 1, 0)].permute(0, 3, 1, 2)  # bgr -> rgb
        # print(img_tensor.shape)
        src_size = tuple(img_tensor.shape[2:])
        
        resize_bool = False
        height, width = img_tensor.shape[2], img_tensor.shape[3]
        if height > 1024 or width > 1024:
            if height > width:
                new_height = 1024
                new_width = int((width / height) * 1024)
            else:
                new_width = 1024
                new_height = int((height / width) * 1024)
            resize_bool = True
            img_tensor = F.interpolate(img_tensor, size=[new_height, new_width], mode='bilinear', align_corners=False)
        else:
            img_tensor = img_tensor
        
        size = tuple(img_tensor.shape[2:])
        
        # inference
        t0 = time.time()
        preds = trt_rmbg(img_tensor)
        print(f"inference use time: {(time.time()-t0):.3f} s")
        if resize_bool:
            size = src_size
            preds = F.interpolate(preds, size=src_size, mode='bilinear', align_corners=False)
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
        