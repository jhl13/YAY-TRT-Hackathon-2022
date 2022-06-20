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
    FirstLayerNormNode = None

    nFill = 0
    nWindowsMask = 0
    nLayerNorm = 0
    nReshapeIn2 = 0
    nRoll = 0
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

        if node.op == "Reshape" and (node.outputs[0].name == "outputs" or node.o().op == "Conv") and ConvNode is not None:
            ReshapeIn2N = gs.Node("ReshapeIn2", "ReshapeIn2-" + str(nReshapeIn2), inputs=[node.inputs[0], ConvNode.outputs[0]], outputs=[node.outputs[0]])
            graph.nodes.append(ReshapeIn2N)
            nReshapeIn2 += 1
            node.outputs = []

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
            if FirstLayerNormNode is None:
                FirstLayerNormNode = layerNormN

        try:
            if node.op == "Reshape" and node.o().op == "Slice" and \
                node.o().o().op == "Concat" and node.o().o().o().op == "Slice" and \
                node.o().o().o().o().op == "Concat":
                inputTensor = node.outputs[0]
                midNode = node.o().o()
                lastNode = node.o().o().o().o()
                tmpNode1 = node.o().o().o().o().o().o()
                tmpNode2 = node.o().o().o().o().o().o().o().o()
                RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[inputTensor], outputs=[midNode.outputs[0]], attrs={"shift": -4, "direction":0})
                graph.nodes.append(RollN)
                nRoll += 1
                RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[midNode.outputs[0]], outputs=[lastNode.outputs[0]], attrs={"shift": -4, "direction":1})
                graph.nodes.append(RollN)
                nRoll += 1
                midNode.outputs = []
                lastNode.outputs = []

                RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[RollN.outputs[0]], outputs=[tmpNode1.outputs[0]], attrs={"shift": 4, "direction":0})
                graph.nodes.append(RollN)
                nRoll += 1
                RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[tmpNode1.outputs[0]], outputs=[tmpNode2.outputs[0]], attrs={"shift": 4, "direction":1})
                graph.nodes.append(RollN)
                nRoll += 1
                tmpNode1.outputs = []
                tmpNode2.outputs = []
        except:
            pass

    if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
        img_mask = ConstantOfShapeNode.outputs[0]
        img_mask_shape = ShapeNode.outputs[0]
        WindowsMaskN = gs.Node("WindowsMask", "WindowsMask-" + str(nWindowsMask), inputs=[img_mask, img_mask_shape], outputs=[ScatterNDNode.outputs[0]])
        graph.nodes.append(WindowsMaskN)
        nWindowsMask += 1
        ScatterNDNode.outputs = []

    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        # ReshapeIn2 挺慢的
        if node.op == "Reshape" and len(node.outputs) > 0 and node.o().op == "Roll":
            ReshapeIn2N = gs.Node("ReshapeIn2", "ReshapeIn2-" + str(nReshapeIn2), inputs=[node.inputs[0], ConvNode.outputs[0]], outputs=[node.outputs[0]])
            graph.nodes.append(ReshapeIn2N)
            nReshapeIn2 += 1
            node.outputs = []
        
        # if node.op == "Roll" and node.o().op == "Reshape":
        #     reshapeN = node.o()
        #     ReshapeIn2N = gs.Node("ReshapeIn2", "ReshapeIn2-" + str(nReshapeIn2), inputs=[reshapeN.inputs[0], FirstLayerNormNode.outputs[0]], outputs=[reshapeN.outputs[0]])
        #     graph.nodes.append(ReshapeIn2N)
        #     nReshapeIn2 += 1
        #     reshapeN.outputs = []
        
        # if node.op == "LayerNorm" and node.o().op == "Reshape":
        #     reshapeN = node.o()
        #     ReshapeIn2N = gs.Node("ReshapeIn2", "ReshapeIn2-" + str(nReshapeIn2), inputs=[reshapeN.inputs[0], ConvNode.outputs[0]], outputs=[reshapeN.outputs[0]])
        #     graph.nodes.append(ReshapeIn2N)
        #     nReshapeIn2 += 1
        #     reshapeN.outputs = []

    print(f"nFill: {nFill}")
    print(f"nWindowsMask: {nWindowsMask}")
    print(f"nLayerNorm: {nLayerNorm}")
    print(f"nReshapeIn2: {nReshapeIn2}")
    print(f"nRoll: {nRoll}")

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
