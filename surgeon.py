import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def surgeon(onnx_path):
    # 读取 .onnx 并进行调整
    graph = gs.import_onnx(onnx.load(onnx_path))

    ConstantOfShapeNode = None
    ShapeNode = None
    ScatterNDNode = None
    ConvNode = None
    ReshapeNode = None

    nFill = 0
    nWindowsMask = 0
    nLayerNorm = 0
    nReshapeIn2 = 0
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

        if node.name == "ConstantOfShape_117": # ConstantOfShape_62 ConstantOfShape_117
            ConstantOfShapeNode = node
        if node.name == "Shape_139": # Shape_84 Shape_139
            ShapeNode = node
        if node.name == "ScatterND_1080": # ScatterND_1025 ScatterND_1080
            ScatterNDNode = node
        if node.name == "Conv_50":
            ConvNode = node
        if node.name == "Reshape_99": # Reshape_2521 Reshape_99
            ReshapeNode = node

        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and \
            node.o().o(0).o().o().o().o() == node.o().o(1) and \
            node.o().o(0).o().o().o().o().o().op == "Mul" and \
            node.o().o(0).o().o().o().o().o().o().op == "Add":

            inputTensor = node.inputs[0]
            MulNode = node.o().o(0).o().o().o().o().o()
            AddNode = node.o().o(0).o().o().o().o().o().o()

            layerNormN = gs.Node("LayerNorm", "LayerNorm-" + str(nLayerNorm), inputs=[inputTensor, MulNode.inputs[1], AddNode.inputs[1]], outputs=[AddNode.outputs[0]])
            graph.nodes.append(layerNormN)
            nLayerNorm += 1
            AddNode.outputs = []

    if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
        img_mask = ConstantOfShapeNode.outputs[0]
        img_mask_shape = ShapeNode.outputs[0]
        WindowsMaskN = gs.Node("WindowsMask", "WindowsMask-" + str(nWindowsMask), inputs=[img_mask, img_mask_shape], outputs=[ScatterNDNode.outputs[0]])
        graph.nodes.append(WindowsMaskN)
        nWindowsMask += 1
        ScatterNDNode.outputs = []

    if ConvNode is not None and ReshapeNode is not None:
        ReshapeIn2N = gs.Node("ReshapeIn2", "ReshapeIn2-" + str(nReshapeIn2), inputs=[ReshapeNode.inputs[0], ConvNode.outputs[0]], outputs=[ReshapeNode.outputs[0]])
        graph.nodes.append(ReshapeIn2N)
        nReshapeIn2 += 1
        ReshapeNode.outputs = []

    print(f"nFill: {nFill}")
    print(f"nWindowsMask: {nWindowsMask}")
    print(f"nLayerNorm: {nLayerNorm}")
    print(f"nReshapeIn2: {nReshapeIn2}")

    graph.cleanup().toposort()
    surgeon_onnx_path = onnx_path.replace(".onnx", "_surgeon.onnx")
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print(f"surgeon model nodes: {len(graph.nodes)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx",
                        help="onnx file path.")
    args = parser.parse_args()
    surgeon(args.onnxFile)
