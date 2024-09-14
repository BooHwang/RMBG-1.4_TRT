# RMBG-1.4_TRT

<div align="center">
<p>Use RMBG-1.4 For Image Matting</p>
</div>

![demo](result/demo.png)

## Usage

Because [BiRefNet](https://github.com/ZhengPeng7/BiRefNet.git) not support TensorRT low than version of 10.2, so this repo will be replace to do matting mission, but this method not good as BiRefNet.

1. export pth to onnx model

```bash
# download weight from hugging face(commit: 22532afbdabdc36b2d30a334076720ac72a06f83)
git clone https://huggingface.co/briaai/RMBG-1.4

python export_onnx.py
```

2. export onnx to tensorrt engine (had test in tensorrt 8.5 and 10.2)

```bash
/root/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/bin/trtexec --onnx=RMBG_1.4.opt.onnx --minShapes=img:1x3x256x256 --optShapes=img:3x3x1024x1024 --maxShapes=img:5x3x1024x1024 --fp16 --saveEngine=RMBG_1.4.plan
```

3. run inference with tensorrt

```bash
# run about 6ms in nvidia l20
python trt_inference.py
```

4. you can compare the inference results with PyTorch's

```bash
# run about 14ms in nvidia l20
python pytorch_inference.py
```