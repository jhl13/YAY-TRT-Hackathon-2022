## 总述  
**队伍名称**：摇阿摇  
**复赛优化模型**：[SwinIR-测试](https://github.com/JingyunLiang/SwinIR)  
上述链接中只包含测试代码，如果需要训练代码，请查看[SwinIR-训练](https://github.com/cszn/KAIR/blob/master/docs/README_SwinIR.md)  
在复赛过程中我们只对训练好的模型进行优化加速，所以只使用测试代码就足够了  

## 原始模型
### 模型简介
- **用途以及效果**  
  SwinIR是一个图像增强模型，可用于图像超分辨率、图像去噪以及图像压缩，且在前述领域中SwinIR性能达到SOTA  
<br/>

- **业界实际运用情况**  
  工业界尚无明确使用SwinIR的用例。但SwinIR性能优越，模型扩展性强，代码简洁易懂，如果有成熟的部署方案的话，相信会有不少团队会愿意尝试在工业界使用SwinIR  
<br/>

- **模型的整体结构**  
  模型由三个部分组成：浅层特征提取模块，深层特征提取模块，高质量重建模块。其中浅层特征模块和高质量重建模块都是基于卷积构建而成的，而深层特征提取模块是基于Swin-Transformer构建的，是一个CNN和Transformer结合的模型  

### 模型优化的难点

**PyTorch模型导出ONNX模型时，会形成大量的算子**
16.4MB的002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth模型导出ONNX后，ONNX包含了29308个节点。64.2MB的001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth模型导出ONNX后，ONNX包含了43819个节点。而初赛中，136MB的encoder模型导出ONNX后，只有1990个节点。大量的节点导致导出ONNX时很容易出现显存不足的问题，同时也不利于ONNX模型的可视化和分析。  

**动态范围更广**
相比于检测/分割等高级语义任务，SwinIR是一个用于低级语义任务的模型，其动态推理要求更高。如检测任务中，常见的动态推理通常只限于batch层面的动态，而输入图片尺寸H/W是固定的。但是在超分、去噪、JPEG压缩等任务中，往往不能提前将输入图片resize成某一固定尺寸，而是要求原图尺寸输入，所以低级语义任务中，输入尺寸H/W也需要是动态的。因为推理过程中需要动态获取H/W信息，所以这也是产生额外shape相关节点的原因之一。  

**转换TRT模型时，涉及形状相关的操作会报错**
SwinIR模型转换为ONNX模型后，产生大量算子的原因有两个：1、计算attention mask需要使用大量的算子；2、模型中存在大量reshape相关的操作，因为动态推理中reshape操作需要在运行过程中获取shape信息，所以会产生大量的shape相关的算子。而这些涉及到shape相关操作的部分，在ONNX转TRT过程中会报错。  
![menSize](./figs/memSize.png)  
![gather](./figs/gather.png) 

## 优化过程  

**Docker**  
建议使用[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt)  
目前官方docker环境中配置的TensorRT版本为8.2.4，但本项目代码在TensorRT 8.2.4与TensorRT 8.4.1.5中均通过测试。

**安装**  
```bash
apt-get install libgl1-mesa-glx
pip install nvidia-pyindex # 需要单独先安装 nvidia-pyindex
pip install -r requirments.txt 
```


**下载预训练模型**
```bash
cd model_zoo/swinir
chmod +x ./download.sh
# 为了节省时间，只下载部分模型，需要下载其他模型，可以对脚本进行修改
./download.sh
```

**PyTorch测评**
```python
# Classical Image Super-Resolution
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

# Lightweight Image Super-Resolution
python main_test_swinir.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

# Real-World Image Super-Resolution
python main_test_swinir.py --task real_sr --scale 2 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth --folder_lq testsets/RealSRSet+5images

# Color Image Deoising
python main_test_swinir.py --task color_dn --noise 15 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth --folder_gt testsets/McMaster

# JPEG Compression Artifact Reduction
python main_test_swinir.py --task jpeg_car --jpeg 10 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth --folder_gt testsets/classic5
```

**导出ONNX模型**
```python
# Classical Image Super-Resolution
python export.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

# Lightweight Image Super-Resolution
python export.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

# Real-World Image Super-Resolution
python export.py --task real_sr --scale 2 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth --folder_lq testsets/RealSRSet+5images

# Color Image Deoising
python export.py --task color_dn --noise 15 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth --folder_gt testsets/McMaster

# JPEG Compression Artifact Reduction
python export.py --task jpeg_car --jpeg 10 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth --folder_gt testsets/classic5
```

**ONNX surgeon**
```python
# Classical Image Super-Resolution
python surgeon.py --onnxFile ./onnx_zoo/swinir_classical_sr_x2/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.onnx

# Lightweight Image Super-Resolution
python surgeon.py --onnxFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx
```

**导出TensorRT模型**
```python
# Classical Image Super-Resolution
python onnx2trt.py --onnxFile ./onnx_zoo/swinir_classical_sr_x2/001_classicalSR_DF2K_s64w8_SwinIR-M_x2_surgeon.onnx

# Lightweight Image Super-Resolution
python onnx2trt.py --onnxFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx
```

**测试TensorRT模型**(支持动态尺寸)  
```python
# Classical Image Super-Resolution
python testTRT.py --onnxFile ./onnx_zoo/swinir_classical_sr_x2/001_classicalSR_DF2K_s64w8_SwinIR-M_x2 --TRTFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.plan

# Lightweight Image Super-Resolution
python testTRT.py --onnxFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx --TRTFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.plan
```


**分步测试模型** 
```python
python3 main_test_swinir.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

python3 export.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

python3 surgeon.py --onnxFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx

python3 onnx2trt.py --onnxFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx

python3 testTRT.py --onnxFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.onnx --TRTFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2_surgeon.plan
```

## 精度与加速效果
无

## Bug报告（可选）
无

## 经验与体会（可选）
无