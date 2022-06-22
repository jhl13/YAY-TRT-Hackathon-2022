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
    nSTReshape = 0
    nRoll = 0
    nSTReshapeRoll = 0
    for node_id, node in enumerate(graph.nodes):
        if node.name == "ConstantOfShape_117": # ConstantOfShape_62 ConstantOfShape_117
            ConstantOfShapeNode = node
        if node.name == "Shape_139": # Shape_84 Shape_139
            ShapeNode = node
        if node.name == "ScatterND_1080": # ScatterND_1025 ScatterND_1080
            ScatterNDNode = node
        if node.name == "Conv_50":
            ConvNode = node

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

    graph.cleanup().toposort()

    # if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
    #     img_mask = ConstantOfShapeNode.outputs[0]
    #     img_mask_shape = ShapeNode.outputs[0]
    #     WindowsMaskN = gs.Node("WindowsMask", "WindowsMask-" + str(nWindowsMask), inputs=[img_mask, img_mask_shape], outputs=[ScatterNDNode.outputs[0]])
    #     graph.nodes.append(WindowsMaskN)
    #     nWindowsMask += 1
    #     ScatterNDNode.outputs = []
    # graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        # if node.op == "LayerNorm" and node.o().op == "Reshape" and len(node.o().outputs) > 0 and \
        #     node.o().outputs[0].name != "outputs" and node.o().o().op == "Slice" and \
        #     node.o().o().o().op == "Concat" and node.o().o().o().o().op == "Slice" and \
        #     node.o().o().o().o().o().op == "Concat":
        #     inputTensor = node.o().outputs[0]
        #     midNode = node.o().o().o()
        #     lastNode = node.o().o().o().o().o()
        #     RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[inputTensor], outputs=[midNode.outputs[0]], attrs={"shift": -4, "direction":0})
        #     graph.nodes.append(RollN)
        #     nRoll += 1
        #     RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[midNode.outputs[0]], outputs=[lastNode.outputs[0]], attrs={"shift": -4, "direction":1})
        #     graph.nodes.append(RollN)
        #     nRoll += 1
        #     midNode.outputs = []
        #     lastNode.outputs = []

        # if node.op == "Transpose" and node.o().op == "Reshape" and len(node.o().outputs) > 0 and \
        #     node.o().outputs[0].name != "outputs" and node.o().o().op == "Slice" and \
        #     node.o().o().o().op == "Concat" and node.o().o().o().o().op == "Slice" and \
        #     node.o().o().o().o().o().op == "Concat":
        #     inputTensor = node.o().outputs[0]
        #     midNode = node.o().o().o()
        #     lastNode = node.o().o().o().o().o()
        #     RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[inputTensor], outputs=[midNode.outputs[0]], attrs={"shift": 4, "direction":0})
        #     graph.nodes.append(RollN)
        #     nRoll += 1
        #     RollN = gs.Node("Roll", "Roll-" + str(nRoll), inputs=[midNode.outputs[0]], outputs=[lastNode.outputs[0]], attrs={"shift": 4, "direction":1})
        #     graph.nodes.append(RollN)
        #     nRoll += 1
        #     midNode.outputs = []
        #     lastNode.outputs = []

        # without shift
        if node.op == "LayerNorm" and node.outputs[0].name != "outputs" and \
                node.o().op == "Reshape" and node.o().o().op != "Slice" and node.o().o(4).op == "Reshape":
            reshapeN = node.o()
            LastN = node.o().o(4)
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[reshapeN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":0, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

        # without shift 可不用
        if node.op == "Reshape" and len(node.outputs) > 0  and node.o().op == "Reshape" and \
                node.o().o().op == "Add" and node.o().o().o(1).op == "LayerNorm":
            reshapeN = node
            LastN = node.o()
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[reshapeN.inputs[0], FirstLayerNormNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":1, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

        # shift
        if node.op == "LayerNorm" and node.o().op == "Reshape"  and \
                len(node.o().outputs) > 0 and node.o().o().op == "Slice" and \
                node.o().o().o().op == "Concat" and node.o().o().o().o().op == "Slice" and \
                node.o().o().o().o().o().op == "Concat" and node.o().o().o().o().o().o(4).op == "Reshape":
            reshapeN = node.o()
            LastN = node.o().o().o().o().o().o(4)
            STReshapeRollN = gs.Node("STReshapeRoll", "STReshapeRoll-" + str(nSTReshapeRoll), 
                                    inputs=[reshapeN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":0, "window_size":8, "shift": -4})
            graph.nodes.append(STReshapeRollN)
            nSTReshapeRoll += 1
            LastN.outputs = []

        # shift 可不用
        if node.op == "Reshape" and len(node.outputs) > 0 and node.o().op == "Slice" and \
                node.o().o().op == "Concat" and node.o().o().o().op == "Slice" and \
                node.o().o().o().o().op == "Concat" and node.o().o().o().o().o().op == "Reshape" and \
                node.o().o().o().o().o().o().op == "Add" and node.o().o().o().o().o().o().o(1).op == "LayerNorm":
            reshapeN = node
            LastN = node.o().o().o().o().o()
            STReshapeRollN = gs.Node("STReshapeRoll", "STReshapeRoll-" + str(nSTReshapeRoll), 
                                    inputs=[reshapeN.inputs[0], FirstLayerNormNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":1, "window_size":8, "shift": 4})
            graph.nodes.append(STReshapeRollN)
            nSTReshapeRoll += 1
            LastN.outputs = []
    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        # 可不用
        if node.op == "Transpose" and node.o().op == "Reshape" and node.o().o().op == "Reshape":
            FirstN = node.o()
            LastN = node.o().o()
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[FirstN.inputs[0], FirstLayerNormNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":2, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

        if node.op == "Reshape" and len(node.outputs) > 0 and \
                node.o().op == "Reshape" and len(node.o().outputs) > 0 and node.o().o().op == "Transpose":
            FirstN = node
            LastN = node.o()
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[FirstN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":3, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []
        
        

    print(f"nFill: {nFill}")
    print(f"nWindowsMask: {nWindowsMask}")
    print(f"nLayerNorm: {nLayerNorm}")
    print(f"nSTReshape: {nSTReshape}")
    print(f"nRoll: {nRoll}")
    print(f"nSTReshapeRoll: {nSTReshapeRoll}")

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
