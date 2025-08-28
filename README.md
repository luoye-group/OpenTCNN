# OpenTCNN

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Chinese](https://img.shields.io/badge/README-中文-blue)](README.md)
[![Endlish](https://img.shields.io/badge/README-English-blue)](README_English.md)
![GitHub repo file or directory count](https://img.shields.io/github/repo-size/luoye-group/OpenTCNN)


![GitHub stars](https://img.shields.io/github/stars/luoye-group/OpenTCNN?style=social)
![GitHub forks](https://img.shields.io/github/forks/luoye-group/OpenTCNN?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/luoye-group/OpenTCNN?style=social)



![GitHub language count](https://img.shields.io/github/languages/count/luoye-group/OpenTCNN)
![GitHub top language](https://img.shields.io/github/languages/top/luoye-group/OpenTCNN)
![GitHub last commit](https://img.shields.io/github/last-commit/luoye-group/OpenTCNN?color=red)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/luoye-group/OpenTCNN)



<div align=center><img src="./logo.png" width="300" ></div>


OpenTCNN 是一个用于构建、训练以及应用热带卷积神经网络 (Tropical CNN) 的开源项目，也是论文 [Compound and Parallel Modes of Tropical Convolutional Neural Networks](https://arxiv.org/abs/2504.06881) 的实现。该项目通过模块化设计和高性能实现，支持多种卷积层类型，并提供丰富的实验案例和实用的工具，便于用户快速上手和扩展。

## 项目特点

- **模块化设计**：网络架构、卷积层和数据预处理均以模块化方式实现，便于扩展与定制。
- **多种卷积层支持**：内置了包括 MinPlus-Sum、MaxPlus-Sum、MinPlus-Max、MaxPlus-Min、MinPlus-Min、MaxPlus-Max 等多种热带卷积层，以及相应的 compound / parallel 变体。详细列表请参考 [README.md](README.md#卷积层支持)。
- **实验丰富**：提供了 conv1d、conv2d 和 conv3d 等多种维度下的实验案例，在 [experiment](experiment/) 目录中均有对应 Jupyter Notebook 示例。
- **数据集模拟**：在 [tcnn/utils/simulation/dataset](tcnn/utils/simulation/dataset/) 中实现了多种数据集处理方法，如 HeifeiECGDataset 等，支持数据增强和重采样。

## 安装指南

1. 克隆仓库：
    ```bash
    git clone https://github.com/nittup/OpenTCNN.git
    cd OpenTCNN
    ```

2. 安装依赖：
    ```bash
    pip install -r requirement.txt
    ```

3. 安装包：
    ```bash
    python setup.py install
    ```

## 快速开始

以下示例展示了如何使用热带卷积层替换传统的卷积层，调用方式与 PyTorch 的卷积层类似。查看 README.md 获取更多示例。

```python
import torch.nn as nn
import tcnn.layers as tlayers

# 传统的卷积层定义
conv = nn.Conv2d(3, 6, kernel_size=5, padding=2)

# 使用热带卷积层
conv = tlayers.MinPlusSumConv2d(3, 6, kernel_size=5, padding=2)
conv = tlayers.MaxPlusSumConv2d(3, 6, kernel_size=5, padding=2)

# 复合卷积层示例：单参数与双参数版本
conv = tlayers.CompoundMinMaxPlusSumConv2d1p(3, 6, kernel_size=5, padding=2)
conv = tlayers.CompoundMinMaxPlusSumConv2d2p(3, 6, kernel_size=5, padding=2)

# 并行卷积层示例
conv = tlayers.ParallelMinMaxPlusSumConv2d1p(3, 6, kernel_size=5, padding=2)
conv = tlayers.ParallelMinMaxPlusSumConv2d2p(3, 6, kernel_size=5, padding=2)
```

## 实验和测试
### 实验
对于论文涉及的实验，请参考 [experiment](experiment) 目录。在此目录下您可以找到基于 conv1d、conv2d 和 conv3d 的多个实验案例，如 `repeat_LeNet_Urban8k.ipynb`。

### 单元测试
项目中的测试位于 tests，您可以通过以下命令运行所有测试用例：
```bash
python -m unittest discover -s tests
```

## 数据集与数据增强
项目内置了一些数据处理和增强方法，如在 `hefei_ecg.py` 中实现的 `HeifeiECGDataset` 类支持 CSV 数据的读取、重采样、数据增强（缩放、翻转、平移）等常用操作。

## 卷积层支持
目前支持的卷积层有：
- Conv1d、Conv2d、Conv3d
- MaxPlusMaxConv1d、MaxPlusMaxConv2d、MaxPlusMaxConv3d
- MaxPlusMinConv1d、MaxPlusMinConv2d、MaxPlusMinConv3d
- MaxPlusSumConv1d、MaxPlusSumConv2d、MaxPlusSumConv3d
- MinPlusMaxConv1d、MinPlusMaxConv2d、MinPlusMaxConv3d
- MinPlusMinConv1d、MinPlusMinConv2d、MinPlusMinConv3d
- MinPlusSumConv1d、MinPlusSumConv2d、MinPlusSumConv3d
- CompoundMinMaxPlusSumConv1d1p、CompoundMinMaxPlusSumConv2d1p、CompoundMinMaxPlusSumConv3d1p
- CompoundMinMaxPlusSumConv1d2p、CompoundMinMaxPlusSumConv2d2p、CompoundMinMaxPlusSumConv3d2p
- ParallelMinMaxPlusSumConv1d1p、ParallelMinMaxPlusSumConv2d1p、 ParallelMinMaxPlusSumConv3d1p
- ParallelMinMaxPlusSumConv1d2p、ParallelMinMaxPlusSumConv2d2p、ParallelMinMaxPlusSumConv3d2p
- ConstantCompoundMinMaxPlusSumConv1d、ConstantCompoundMinMaxPlusSumConv2d、ConstantCompoundMinMaxPlusSumConv3d
- ConstantParallelMinMaxPlusSumConv1d、ConstantParallelMinMaxPlusSumConv2d、ConstantParallelMinMaxPlusSumConv3d

## 贡献

欢迎对 OpenTCNN 做出贡献！

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 引用 OpenTCNN
如果您在研究中使用了 OpenTCNN，请考虑引用OpenTCNN 或者 我们的论文：
- 论文：
```bibtex
@misc{li2025compoundparallelmodestropical,
      title={Compound and Parallel Modes of Tropical Convolutional Neural Networks}, 
      author={Mingbo Li and Liying Liu and Ye Luo},
      year={2025},
      eprint={2504.06881},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.06881}, 
}
```
- OpenTCNN：
```bibtex
@software{opentcnn2025,
   author = {{Mingbo Li, Liying Liu and Ye Luo}},
   month = {4},
   title = {{OpenTCNN}},
   url = {https://github.com/luoye-group/OpenTCNN},
   version = {1.0},
   year = {2025}
}
```

