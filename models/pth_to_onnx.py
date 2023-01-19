import sys
sys.path.append("/workspace/tensorrt/")

from src.esrgan import ESRGAN
import torch

print("Converting")
model_name = sys.argv[1].strip()

model = ESRGAN(f"/workspace/tensorrt/models/{model_name}.pth")
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
