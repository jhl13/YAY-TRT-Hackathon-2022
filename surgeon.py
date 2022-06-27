import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def surgeon(args):
    onnx_path = args.onnxFile
    window_size = 8
    if args.task == "jpeg_car":
        window_size = 7
    # 读取 .onnx 并进行调整
    graph = gs.import_onnx(onnx.load(onnx_path))

    ConstantOfShapeNode = None
    ShapeNode = None
    ScatterNDNode = None
    ConvNode = None
    FirstLayerNormNode = None
    FirstSTReshapeNode = None

    nFill = 0
    nWindowsMask = 0
    nLayerNorm = 0
    nSTReshape = 0
    nSTReshapeRoll = 0
    nSTReshapeAdd = 0
    nMyGather = 0
    for node_id, node in enumerate(graph.nodes):
        try:
            if node.op == "ConstantOfShape" and node.o(5).op == "Shape":
                ConstantOfShapeNode = node
                ShapeNode = node.o(5)
        except:
            pass
        if node.op == "ScatterND":
            ScatterNDNode = node
        if node.op == "Conv" and ConvNode is None:
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

    if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
        img_mask = ConstantOfShapeNode.outputs[0]
        img_mask_shape = ShapeNode.outputs[0]
        WindowsMaskN = gs.Node("WindowsMask", "WindowsMask-" + str(nWindowsMask), 
                                    inputs=[img_mask, img_mask_shape], 
                                    outputs=[ScatterNDNode.outputs[0]],
                                    attrs={"window_size":window_size})
        graph.nodes.append(WindowsMaskN)
        nWindowsMask += 1
        ScatterNDNode.outputs = []
    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        # without shift
        if node.op == "LayerNorm" and node.outputs[0].name != "outputs" and \
                node.o().op == "Reshape" and node.o().o().op != "Slice" and node.o().o(4).op == "Reshape":
            reshapeN = node.o()
            LastN = node.o().o(4)
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[reshapeN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":0, "window_size":window_size, "num_heads":6})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

        # # without shift 可不用
        # if node.op == "Reshape" and len(node.outputs) > 0  and node.outputs[0].name != "outputs" and node.o().op == "Reshape" and \
        #         node.o().o().op == "Add" and node.o().o().o(1).op == "LayerNorm":
        #     reshapeN = node
        #     LastN = node.o()
        #     STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
        #                             inputs=[reshapeN.inputs[0], FirstLayerNormNode.outputs[0]], 
        #                             outputs=[LastN.outputs[0]],
        #                             attrs={"type":1, "window_size":window_size, "num_heads":6})
        #     graph.nodes.append(STReshapeN)
        #     nSTReshape += 1
        #     LastN.outputs = []

        # shift
        if node.op == "LayerNorm" and node.outputs[0].name != "outputs" and node.o().op == "Reshape"  and \
                len(node.o().outputs) > 0 and node.o().o().op == "Slice" and \
                node.o().o().o().op == "Concat" and node.o().o().o().o().op == "Slice" and \
                node.o().o().o().o().o().op == "Concat" and node.o().o().o().o().o().o(4).op == "Reshape":
            reshapeN = node.o()
            LastN = node.o().o().o().o().o().o(4)
            STReshapeRollN = gs.Node("STReshapeRoll", "STReshapeRoll-" + str(nSTReshapeRoll), 
                                    inputs=[reshapeN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":0, "window_size":window_size, "shift": -4})
            graph.nodes.append(STReshapeRollN)
            nSTReshapeRoll += 1
            LastN.outputs = []

        # shift
        if node.op == "Reshape" and len(node.outputs) > 0 and node.outputs[0].name != "outputs" and node.o().op == "Slice" and \
                node.o().o().op == "Concat" and node.o().o().o().op == "Slice" and \
                node.o().o().o().o().op == "Concat" and node.o().o().o().o().o().op == "Reshape" and \
                node.o().o().o().o().o().o().op == "Add" and node.o().o().o().o().o().o().o(1).op == "LayerNorm":
            reshapeN = node
            LastN = node.o().o().o().o().o()
            STReshapeRollN = gs.Node("STReshapeRoll", "STReshapeRoll-" + str(nSTReshapeRoll), 
                                    inputs=[reshapeN.inputs[0], FirstLayerNormNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":1, "window_size":window_size, "shift": 4})
            graph.nodes.append(STReshapeRollN)
            nSTReshapeRoll += 1
            LastN.outputs = []
    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        # # 可不用
        # if node.op == "Transpose" and node.o().op == "Reshape" and \
        #     len(node.o().outputs) > 0 and node.o().outputs[0].name != "outputs" and \
        #     node.o().o().op == "Reshape" and node.o().o().o().op != "Unsqueeze":
        #     FirstN = node.o()
        #     LastN = node.o().o()
        #     STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
        #                             inputs=[FirstN.inputs[0], FirstLayerNormNode.outputs[0]], 
        #                             outputs=[LastN.outputs[0]],
        #                             attrs={"type":2, "window_size":window_size, "num_heads":6})
        #     graph.nodes.append(STReshapeN)
        #     nSTReshape += 1
        #     LastN.outputs = []

        if node.op == "Reshape" and len(node.outputs) > 0 and node.outputs[0].name != "outputs" and \
                node.o().op == "Reshape" and len(node.o().outputs) > 0 and node.o().o().op == "Transpose":
            FirstN = node
            LastN = node.o()
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[FirstN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":3, "window_size":window_size, "num_heads":6})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []
    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        if node.op == "Reshape" and node.o().op == "Conv":
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[node.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[node.outputs[0]],
                                    attrs={"type":4, "window_size":window_size, "num_heads":6})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            node.outputs = []
    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        if node.op == "LayerNorm" and node.o().op == "STReshape" and \
            node.o().o().op == "Transpose" and node.o().o().o().op == "STReshape":
            FirstSTReshapeNode = node.o().o().o()
            break
        if node.op == "Transpose" and node.o().op == "Reshape" and \
            len(node.o().outputs) > 0 and node.o().outputs[0].name != "outputs" and \
            node.o().o().op == "Reshape" and node.o().o().o().op != "Unsqueeze":
            FirstSTReshapeNode = node.o().o()
            break

    for node_id, node in enumerate(graph.nodes):
        # # 可不用
        # if node.op == "STReshape" and node.o().op == "Shape" and node.o(3).op == "MatMul" and node.o(3).o().op == "Add" and \
        #     node.o(3).o().o().op == "Reshape":
        #     reshapeNode = node.o(3).o().o()
        #     STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
        #                             inputs=[reshapeNode.inputs[0], FirstSTReshapeNode.outputs[0]], 
        #                             outputs=[reshapeNode.outputs[0]],
        #                             attrs={"type":5, "num_heads":6, "window_size":window_size})
        #     graph.nodes.append(STReshapeN)
        #     nSTReshape += 1
        #     reshapeNode.outputs = []

        # # 可不用
        # if node.op == "Softmax" and node.o().op == "MatMul" and node.o().o().op == "Transpose" and node.o().o().o().op == "Reshape":
        #     reshapeNode = node.o().o().o()
        #     STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
        #                             inputs=[reshapeNode.inputs[0], FirstSTReshapeNode.outputs[0]], 
        #                             outputs=[reshapeNode.outputs[0]],
        #                             attrs={"type":6, "num_heads":6, "window_size":window_size})
        #     graph.nodes.append(STReshapeN)
        #     nSTReshape += 1
        #     reshapeNode.outputs = []

        if node.op == "Mul" and node.o().op == "MatMul" and node.o().o().op == "Add" and \
            node.o().o().o().op == "Reshape" and node.o().o().o().o().op == "Add" and \
             node.o().o().o().o().o().op == "Reshape":
            FirstNode = node.o().o().o()
            LastNode = node.o().o().o().o().o()
            positionNode = node.o().o()
            maskNode = node.o().o().o().o()
            STReshapeAddN = gs.Node("STReshapeAdd", "STReshapeAdd-" + str(nSTReshapeAdd), 
                                    inputs=[positionNode.inputs[0], positionNode.inputs[1], maskNode.inputs[1]], 
                                    outputs=[LastNode.outputs[0]],
                                    attrs={"type":6, "num_heads":6, "window_size": window_size})
            graph.nodes.append(STReshapeAddN)
            nSTReshapeAdd += 1
            LastNode.outputs = []
    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        try:
            if node.op == "Transpose" and node.o().op == "Gather" and node.o().o().op == "Mul" and node.o().o().o().op == "MatMul" and \
                node.o(1).op == "Gather" and node.o(1).o().op == "Transpose" and node.o(1).o().o().op == "MatMul" and \
                node.o(2).op == "Gather" and node.o(2).o().op == "MatMul" and node.o(2).o().o().op == "Transpose":
                FirstNode = node
                Gather1Node = node.o()
                Gather2Node = node.o(1)
                Gather3Node = node.o(2)
                MulNode = node.o().o()
                MyGatherN = gs.Node("MyGather", "MyGather-" + str(nMyGather), 
                                        inputs=[FirstNode.outputs[0], MulNode.inputs[1]], 
                                        outputs=[MulNode.outputs[0], Gather2Node.outputs[0], Gather3Node.outputs[0]])
                graph.nodes.append(MyGatherN)
                nMyGather += 1
                MulNode.outputs = []
                Gather2Node.outputs = []
                Gather3Node.outputs = []
        except:
            pass


    print(f"nFill: {nFill}")
    print(f"nWindowsMask: {nWindowsMask}")
    print(f"nLayerNorm: {nLayerNorm}")
    print(f"nSTReshape: {nSTReshape}")
    print(f"nSTReshapeRoll: {nSTReshapeRoll}")
    print(f"nSTReshapeAdd: {nSTReshapeAdd}")
    print(f"nMyGather: {nMyGather}")

    graph.cleanup().toposort()
    surgeon_onnx_path = onnx_path.replace(".onnx", "_surgeon.onnx")
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print(f"surgeon model nodes: {len(graph.nodes)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx",
                        help="onnx file path.")
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()
    surgeon(args)
