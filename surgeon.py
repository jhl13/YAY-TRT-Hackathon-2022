import onnx
import onnx_graphsurgeon as gs
import numpy as np

# 读取 .onnx 并进行调整
graph = gs.import_onnx(onnx.load("/target/SwinIR/onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx"))

nLayerNorm = 0
for node_id, node in enumerate(graph.nodes):
    if node.op == 'ReduceMean' and \
        node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
        node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
        node.o().o(0).o().op == 'ReduceMean' and \
        node.o().o(0).o().o().op == 'Add' and \
        node.o().o(0).o().o().o().op == 'Sqrt' and \
        node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):

        inputTensor = node.inputs[0]
        lastDivNode = node.o().o(0).o().o().o().o()

        layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNorm), inputs=[inputTensor], outputs=[lastDivNode.outputs[0]])
        lastDivNode.outputs = []
        nLayerNorm += 1


print(f"nLayerNorm: {nLayerNorm}") 

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "/target/SwinIR/onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx")
