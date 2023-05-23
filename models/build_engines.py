import re
import subprocess
import sys
sys.path.append("/workspace/tensorrt/")

from glob import glob
from pathlib import Path
from src.esrgan import ESRGAN
from src.SRVGGNetCompact_arch import SRVGGNetCompact
import torch

MODELS_DIRECTORY = "/workspace/tensorrt/models"

def build_engine(onnx_file):
    arguments = [
        "trtexec",
        "--fp16",
        f"--onnx={onnx_file}",
        "--minShapes=input:1x3x540x720",
        "--optShapes=input:1x3x540x720",
        "--maxShapes=input:1x3x540x720",
        f"--saveEngine={MODELS_DIRECTORY}/{Path(onnx_file).stem}.engine",
        "--tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT"
    ]
    subprocess.call(arguments)

def build_onnx(pth_file) -> str:
    model_name = Path(pth_file).stem
    onnx_file = f"/tmp/{model_name}.onnx"

    try:
        model = ESRGAN(pth_file)
    except:
        num_feat = 64
        num_conv = 16
        
        match = re.search(r'nf(\d+)-nc(\d+)', model_name)
        if match:
            num_feat = int(match.group(1))
            num_conv = int(match.group(2))
        
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=int(model_name[0]), act_type="prelu"
        )
        
        state_dict = torch.load(pth_file)

        if "params" in state_dict.keys():
            model.load_state_dict(state_dict["params"])
        else:
            model.load_state_dict(state_dict)
        
    model.eval().cuda()
    # https://github.com/onnx/onnx/issues/654
    dynamic_axes = {
        "input": {0: "batch_size", 2: "width", 3: "height"},
        "output": {0: "batch_size", 2: "width", 3: "height"},
    }
    dummy_input = torch.rand(1, 3, 64, 64).cuda()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        opset_version=14,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    return onnx_file

if __name__ == "__main__":
    for pth_file in glob(f"{MODELS_DIRECTORY}/*.pth"):
        if not Path(pth_file.replace(".pth", ".engine")).is_file():
            print(Path(pth_file).stem)
            onnx_file = build_onnx(pth_file)
            build_engine(onnx_file)

