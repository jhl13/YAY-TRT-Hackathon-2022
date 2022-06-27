import os
import torch
import onnx
import tensorrt as trt
import torch.nn as nn
from glob import glob
import ctypes
import numpy as np
from cuda import cudart
import numpy as np
from time import time_ns

onnx_model = 'model.onnx'
plan_file = 'model.plan'

class NaiveModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(x)


device = torch.device('cuda:0')

# generate ONNX model
torch.onnx.export(
    NaiveModel(),
    torch.randn(1089, 6, 64, 64),
    onnx_model,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=dict(
        input=dict({
            0: 'batch',
        }),
        output=dict({0: 'batch'})),
    opset_version=13)

inputs_random = torch.randn(1089, 6, 64, 64).to(device)
model = NaiveModel().to(device)
print(f"input shape: {inputs_random.shape}")

for i in range(3):
    model(inputs_random)

t0 = time_ns()
for i in range(30):
    outputs = model(inputs_random)
t1 = time_ns()
timePerInference = (t1-t0)/1000/1000/30
print(f"Pytorch softmax time: {timePerInference}")

# load_tensorrt_plugin()
# create builder and network
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# parse onnx
parser = trt.OnnxParser(network, logger)
parser.parse_from_file(onnx_model)

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
profile = builder.create_optimization_profile()

profile.set_shape('input', [361, 6, 64, 64], [361, 6, 64, 64],
                  [1089, 6, 64, 64])
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)

try:
    with open(plan_file, 'wb') as f:
        f.write(engineString)
    print("export .plan successful")
except:
    print("export .plan fail")
# 将没法转换的子图单独保存 


#-------------------------------------------------------------------------------

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
plugin_path = "plugin/"
soFileList = glob(plugin_path + "*.so")

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)
#-------------------------------------------------------------------------------

print("Test Plan!")
if os.path.isfile(plan_file):
    with open(plan_file, 'rb') as encoderF:
        engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
    if engine is None:
        print("Failed loading %s"%plan_file)
        exit()
    print("Succeeded loading %s"%plan_file)
else:
    print("Failed finding %s"%plan_file)
    exit()
nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
nOutput = engine.num_bindings - nInput
context = engine.create_execution_context()
#-------------------------------------------------------------------------------

inputs = np.random.randn(1089, 6, 64, 64)
context.set_binding_shape(0, inputs.shape)
bufferH = []
bufferH.append( inputs.astype(np.float32).reshape(-1) )

for i in range(nInput, nInput + nOutput):                
    bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

bufferD = []
for i in range(nInput + nOutput):                
    bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nInput, nInput + nOutput):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

# warm up
for i in range(10):
    context.execute_v2(bufferD)

# test infernece time
t0 = time_ns()
for i in range(30):
    context.execute_v2(bufferD)
t1 = time_ns()
timePerInference = (t1-t0)/1000/1000/30
print(f"TRT softmax time: {timePerInference}")

index_output = engine.get_binding_index("output")
output = bufferH[index_output]
