import sys
sys.path.append("/workspace/tensorrt/")

from src.esrgan import ESRGAN
from src.SRVGGNetCompact_arch import SRVGGNetCompact
import torch

print("Converting")

model_name = sys.argv[1].strip()
model_path = f"/workspace/tensorrt/models/{model_name}.pth"

try:
    model = ESRGAN(model_path)
except:
    # parameters depend on model and you need to set them manually if it errors
    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=int(model_name[0]), act_type="prelu"
    )
    
    state_dict = torch.load(model_path)

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
    f"/tmp/{model_name}.onnx",
    opset_version=14,
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
)

print("Finished")
