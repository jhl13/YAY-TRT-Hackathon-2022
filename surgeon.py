import onnx
import onnx_graphsurgeon as gs
import numpy as np

# 读取 .onnx 并进行调整
graph = gs.import_onnx(onnx.load("./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx"))

ConstantOfShapeNode = None
ShapeNode = None
ScatterNDNode = None
CastNode = None

nFill = 0
nWindowsMask = 0
for node_id, node in enumerate(graph.nodes):
    if node.op == 'Sub' and node.o().op == "Equal" and \
        node.o().o().op == "Not" and \
        node.o().o().o().op == "Cast" and \
        node.o().o().o().o().op == "Where" and \
        node.o().o().o().o().o().op == "Cast":

        inputTensor = node.outputs[0]
        lastNode = node.o().o().o().o().o()
        FillN = gs.Node("Fill", "Fill-" + str(nFill), inputs=[inputTensor], outputs=[lastNode.outputs[0]])
        graph.nodes.append(FillN)
        lastNode.outputs = []
        nFill += 1

    if node.name == "ConstantOfShape_62":
        ConstantOfShapeNode = node
    if node.name == "Shape_84":
        ShapeNode = node
    if node.name == "ScatterND_1025":
        ScatterNDNode = node
    if node.name == "Cast_63":
        CastNode = node

if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
    img_mask = ConstantOfShapeNode.outputs[0]
    img_mask_shape = ShapeNode.outputs[0]
    WindowsMaskN = gs.Node("WindowsMask", "WindowsMask-" + str(nWindowsMask), inputs=[img_mask, img_mask_shape], outputs=[ScatterNDNode.outputs[0]])
    graph.nodes.append(WindowsMaskN)
    nWindowsMask += 1
    ScatterNDNode.outputs = []

print(f"nFill: {nFill}")
print(f"nWindowsMask: {nWindowsMask}")

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx")
print(f"surgeon model nodes: {len(graph.nodes)}")
