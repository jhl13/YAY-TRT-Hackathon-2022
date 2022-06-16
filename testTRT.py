import onnx
import onnx_graphsurgeon as gs

import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
from collections import OrderedDict
import cv2

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def check(a, b, weak=False, epsilon = 1e-5):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    #print("check:",res,diff0,diff1)
    return res, diff0, diff1


graph = gs.import_onnx(onnx.load("onnx_zoo/swinir_classical_sr_x2/001_classicalSR_DF2K_s64w8_SwinIR-M_x2_surgeon.onnx"))
print("graph nodes: ", len(graph.nodes))

folder_lq = "testsets/Set5/LR_bicubic/X2"
folder_gt = "testsets/Set5/HR"
plan_file = "onnx_zoo/swinir_classical_sr_x2/001_classicalSR_DF2K_s64w8_SwinIR-M_x2_surgeon.plan"
plugin_path = "plugin/"
soFileList = glob(plugin_path + "*.so")
task = "lightweight_sr"
scale = 2
window_size = 8
border = 2

def get_image_pair(task, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if task in ["classical_sr", "lightweight_sr"]:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = cv2.imread(f"{folder_lq}/{imgname}x{scale}{imgext}", cv2.IMREAD_COLOR).astype(
            np.float32) / 255.
    return imgname, img_lq, img_gt
#-------------------------------------------------------------------------------

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

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

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []
test_results['psnr_y'] = []
test_results['ssim_y'] = []
test_results['psnr_b'] = []
psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

for idx, path in enumerate(sorted(glob(os.path.join(folder_gt, '*.png')))):
    imgname, img_lq, img_gt = get_image_pair(task, path)  # image to HWC-BGR, float32
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = img_lq[np.newaxis, :, :, :]

    _, _, h_old, w_old = img_lq.shape
    # if h_old != 256 and w_old != 256:
    #     continue
    # h_pad = (h_old // window_size + 1) * window_size - h_old
    # w_pad = (w_old // window_size + 1) * window_size - w_old
    # img_lq = np.concatenate([img_lq, np.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    # img_lq = np.concatenate([img_lq, np.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

    context.set_binding_shape(0, img_lq.shape)
    bufferH = []
    bufferH.append( img_lq.astype(np.float32).reshape(-1) )

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

    index_output = engine.get_binding_index("outputs")
    output = bufferH[index_output]

    gt_data = np.load(path.replace(".png", ".npz"))
    gt_output = gt_data["output"]
    check_res = check(output, gt_output, True)
    print(output.shape)
    string = "%4d,%4d,%8.3f,%9.3e,%9.3e"%(h_old, w_old, timePerInference, check_res[1], check_res[2])
    print(string + ", %s"%("Good" if check_res[1] < 1e-4 and check_res[2] < 1e-4 else "Bad"))

    # output = output[..., :h_old * scale, :w_old * scale]

    # # save image
    # output = np.clip(np.squeeze(output), 0.0, 1.0)
    # if output.ndim == 3:
    #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    # output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    # # evaluate psnr/ssim/psnr_b
    # if img_gt is not None:
    #     img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
    #     img_gt = img_gt[:h_old * scale, :w_old * scale, ...]  # crop gt
    #     img_gt = np.squeeze(img_gt)

    #     psnr = calculate_psnr(output, img_gt, crop_border=border)
    #     ssim = calculate_ssim(output, img_gt, crop_border=border)
    #     test_results['psnr'].append(psnr)
    #     test_results['ssim'].append(ssim)
    #     if img_gt.ndim == 3:  # RGB image
    #         psnr_y = calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
    #         ssim_y = calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
    #         test_results['psnr_y'].append(psnr_y)
    #         test_results['ssim_y'].append(ssim_y)
    #     print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
    #             'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
    #             'PSNR_B: {:.2f} dB.; Inference time: {:.2f}'.
    #             format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b, timePerInference))
    # else:
    #     print('Testing {:d} {:20s}'.format(idx, imgname))
