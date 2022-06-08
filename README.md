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

**目前遇到的问题**  
1. PyTorch模型导出ONNX模型时，会形成大量的算子  
16.4MB的002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth模型导出ONNX后，ONNX包含了29308个节点/13420-BasicLayer/7462-RSTB/6559-只保留shift-mask计算过程  
64.2MB的001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth模型导出ONNX后，ONNX包含了43819个节点  
而初赛中，136MB的encoder模型导出ONNX后，只有1990个节点  
大量的节点导致导出ONNX时很容易出现显存不足的问题。在导ONNX模型后发现，很多节点其实是涉及形状操作的节点，后续解决思路有两个：1、从源代码出发，删除或合一些形状相关的操作；2、对ONNX网络进行修改。  
<br/>

2. ONNX模型转TRT时，会有尺寸相关报错信息  
初步定位到的问题是原模型保存了attention mask，导致导出ONNX时把mask当成是一个常量，而不是一个随着输入图像尺寸变化而变化的变量。后续解决思路有两个：1、在网络前向时才生成mask；2、提前计算好mask，把mask当成输入。  

## 优化过程  

**安装**  
```bash
apt-get install libgl1-mesa-glx
pip install -r requirments.txt # 有可能需要单独先安装 nvidia-pyindex
```


**下载预训练模型**
```bash
cd model_zoo/swinir
chmod +x ./download.sh
./download.sh
```

**PyTorch测评**
```python
python main_test_swinir.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
```

**导出ONNX模型**
```python
python export.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
```

**导出TensorRT模型**
```python
python onnx2trt.py --onnxFile ./onnx_zoo/swinir_lightweight_sr_x2/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx
```

**测试TensorRT模型**(目前只支持固定尺寸，不支持动态尺寸)  
```python
python testTRT.py
```

## 精度与加速效果
无

## Bug报告（可选）
无

## 经验与体会（可选）
无