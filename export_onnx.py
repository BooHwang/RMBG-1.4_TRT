#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      : export_onnx.py
@Time      : 2024/09/13 20:40:50
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
/root/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/bin/trtexec --onnx=RMBG_1.4.opt.onnx --minShapes=img:1x3x256x256 --optShapes=img:3x3x1024x1024 --maxShapes=img:5x3x1024x1024 --fp16 --saveEngine=RMBG_1.4.plan
'''

import torch
import onnx
from onnxsim import simplify
from pprint import pprint
import onnxruntime

from pytorch_inference import BriaRMBGPipe


def check_model(onnx_path, onnx_opt_path):
    model = onnx.load(onnx_path)
    simplified_model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(simplified_model, onnx_opt_path)
    try:
        onnx.checker.check_model(onnx_opt_path)
        print("model check passed!")
    except Exception as e:
        print(f"model check failed: {e}")
        
def print_in_out_name(onnx_opt_path):
    onnx_model = onnx.load(onnx_opt_path)
    onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    print("------input-------")
    input_tensors = onnx_session.get_inputs()
    for input_tensor in input_tensors:
        input_info = {
            "name": input_tensor.name,
            "type": input_tensor.type,
            "shape": input_tensor.shape,
        }
        pprint(input_info)
        
    print("------output-------")
    output_tensors = onnx_session.get_outputs()
    for output_tensor in output_tensors:
        output_info = {
            "name" : output_tensor.name,
            "type" : output_tensor.type,
            "shape": output_tensor.shape,
        }
        pprint(output_info)
        
        
        
if __name__ == "__main__":
    rmbg = BriaRMBGPipe("RMBG-1.4") # hugging face
    fp16 = False
    
    # export to onnx
    onnx_path = 'RMBG_1.4.onnx'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = torch.randint(0, 256, (1, 3, 1024, 1024), dtype=torch.uint8).to(device)
    onnx_opt_path = onnx_path.replace(".onnx", ".opt.onnx")
    
    torch.onnx.export(
        rmbg.half() if fp16 else rmbg,
        img_tensor,
        onnx_path,
        opset_version=17,
        input_names=['img'],
        output_names=['mask'],
        dynamic_axes={"img": {0: "bs", 2: "h", 3: "w"},
                      "mask": {0: "bs", 2: "h", 3: "w"},
                     },
    )
    check_model(onnx_path, onnx_opt_path)
    print_in_out_name(onnx_opt_path)