# import sys
# print(sys.path)
import onnx
from onnxsim import simplify
from onnx import shape_inference

onnx_file = "./tmp.onnx"
# load your predefined ONNX model
model = onnx.load(onnx_file)

# convert model
model_simp, check = simplify(model)
inferred_onnx_model = shape_inference.infer_shapes(model_simp)
onnx.save(inferred_onnx_model, "shufflenetv2_onnxsim_128_128_73.onnx")
assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object