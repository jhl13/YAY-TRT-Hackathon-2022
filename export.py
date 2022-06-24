import argparse
from ntpath import join
from pathlib import Path
import glob
import numpy as np
import os
import torch
import requests
import onnx
import onnx_graphsurgeon as gs

from main_test_swinir import get_image_pair, define_model


def export():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=48, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default='testsets/Set5/LR_bicubic/X2', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='testsets/Set5/HR', help='input ground-truth test image folder')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    save_dir, window_size = setup(args)
    suffix = Path(args.model_path).suffix
    onnx_name = Path(args.model_path).name.replace(suffix, ".onnx")
    onnx_file = os.path.join(save_dir, onnx_name)
    os.makedirs(save_dir, exist_ok=True)

    x = torch.randn((1, 3, window_size*32, window_size*32), requires_grad=False).to(device)
    torch.onnx.export(model, 
                     (x), 
                     onnx_file, 
                     verbose=False,
                     opset_version=args.opset,
                     do_constant_folding=True,
                     input_names=['images'],
                     output_names=['outputs'],
                     dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},
                                    # 'outputs': {0: 'batch', 1: 'height_out', 2: 'width_out'}
                                    'outputs': {0: 'batch', 2: 'height_out', 3: 'width_out'}
                                  }
                    )

    graph = gs.import_onnx(onnx.load(onnx_file))
    print(f"Number of onnx nodes: {len(graph.nodes)}")

def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'onnx_zoo/swinir_{args.task}_x{args.scale}'
        window_size = 8

    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'onnx_zoo/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'onnx_zoo/swinir_{args.task}_noise{args.noise}'
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ['jpeg_car']:
        save_dir = f'onnx_zoo/swinir_{args.task}_jpeg{args.jpeg}'
        window_size = 7

    return save_dir, window_size


if __name__ == '__main__':
    export()
