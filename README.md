# 基于深度学习的端到端视频压缩算法设计与实现

## 项目简介
本项目是一个基于 PyTorch 的端到端视频压缩框架，旨在探究神经网络在去除视频时空冗余方面的有效性。
核心创新点在于引入了 **CBAM (Convolutional Block Attention Module)** 注意力机制，以提升视频重建的主观视觉质量。

## 目录结构
```
src/
  models/
    attention.py    # CBAM 注意力模块实现 (Scheme A)
    motion.py       # SPyNet 光流估计网络
    compression.py  # 运动压缩与残差压缩网络 (包含 CBAM)
    video_net.py    # 完整的视频压缩模型
  utils/
    metrics.py      # PSNR, MS-SSIM 评价指标
requirements.txt    # 依赖库
train.py            # 训练脚本 (示例)
```

## 环境配置
请确保安装了 Python 3.8+ 和 PyTorch。
```bash
pip install -r requirements.txt
```

## 核心架构
1. **Motion Net**: 使用 SPyNet 估计光流。
2. **Motion Compensation**: 基于光流进行运动补偿。
3. **Residual Net**: 引入 CBAM 的残差压缩网络。
4. **Entropy Model**: 使用 `compressai` 的熵瓶颈层。

## 运行指南
(此处为示例，需根据具体数据集调整)
```bash
python train.py --dataset /path/to/vimeo90k
```

## 评价指标
- **BPP (Bits Per Pixel)**: 压缩率
- **PSNR**: 峰值信噪比
- **MS-SSIM**: 多尺度结构相似性
