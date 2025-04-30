# OpenTCNN

[![Chinese](https://img.shields.io/badge/README-中文-blue)](README.md)
[![Endlish](https://img.shields.io/badge/README-English-blue)](README_English.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![GitHub repo file or directory count](https://img.shields.io/github/repo-size/luoye-group/OpenTCNN)

OpenTCNN is an open source project for building, training, and applying Tropical Convolutional Neural Networks (Tropical CNN). It is also the implementation of the paper [Compound and Parallel Modes of Tropical Convolutional Neural Networks](https://arxiv.org/abs/2504.06881). The project features a modular design and high-performance implementation that supports multiple types of convolutional layers, and provides abundant experimental cases and practical tools to facilitate rapid adoption and extension.

## Key Features

- **Modular Design**: The network architecture, convolutional layers, and data preprocessing are implemented in a modular manner, making it easy to extend and customize.
- **Support for Various Convolutional Layers**: It includes several tropical convolutional layers such as MinPlus-Sum, MaxPlus-Sum, MinPlus-Max, MaxPlus-Min, MinPlus-Min, MaxPlus-Max, and their compound/parallel variants. For more details see [README.md](README.md#convolutional-layer-support).
- **Rich Experiments**: Provided experiments for conv1d, conv2d, and conv3d are available as Jupyter Notebook examples in the [experiment](experiment/) directory.
- **Dataset Simulation**: Multiple dataset processing methods are implemented under [tcnn/utils/simulation/dataset](tcnn/utils/simulation/dataset/), such as the Heifei ECG Dataset class, supporting operations like data augmentation and resampling.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/nittup/OpenTCNN.git
    cd OpenTCNN
    ```

2. Install dependencies:
    ```bash
    pip install -r requirement.txt
    ```

3. Install the package:
    ```bash
    python setup.py install
    ```

## Quick Start

The following example demonstrates how to replace traditional convolutional layers with tropical convolutional layers, which are used similarly to PyTorch’s convolutional layers. For more examples see README.md.

```python
import torch.nn as nn
import tcnn.layers as tlayers

# Traditional convolutional layer
conv = nn.Conv2d(3, 6, kernel_size=5, padding=2)

# Using tropical convolutional layers
conv = tlayers.MinPlusSumConv2d(3, 6, kernel_size=5, padding=2)
conv = tlayers.MaxPlusSumConv2d(3, 6, kernel_size=5, padding=2)

# Example: Compound convolutional layers (single-parameter and two-parameter versions)
conv = tlayers.CompoundMinMaxPlusSumConv2d1p(3, 6, kernel_size=5, padding=2)
conv = tlayers.CompoundMinMaxPlusSumConv2d2p(3, 6, kernel_size=5, padding=2)

# Example: Parallel convolutional layers
conv = tlayers.ParallelMinMaxPlusSumConv2d1p(3, 6, kernel_size=5, padding=2)
conv = tlayers.ParallelMinMaxPlusSumConv2d2p(3, 6, kernel_size=5, padding=2)
```

## Experiments and Testing
### Experiments
For the experiments discussed in the paper, please refer to the [experiment](experiment) directory where you can find several experiment cases based on conv1d, conv2d, and conv3d (e.g. `repeat_LeNet_Urban8k.ipynb`).

### Unit Tests
The tests are located in the tests directory. You can run all test cases with:
```bash
python -m unittest discover -s tests
```

## Datasets and Data Augmentation
The project includes several data processing and augmentation methods. For example, the `HeifeiECGDataset` class in `hefei_ecg.py` supports reading CSV data, resampling, and common augmentation techniques such as scaling, flipping, and translating.

## Convolutional Layer Support
Currently, the following convolutional layers are supported:
- Conv1d, Conv2d, Conv3d
- MaxPlusMaxConv1d, MaxPlusMaxConv2d, MaxPlusMaxConv3d
- MaxPlusMinConv1d, MaxPlusMinConv2d, MaxPlusMinConv3d
- MaxPlusSumConv1d, MaxPlusSumConv2d, MaxPlusSumConv3d
- MinPlusMaxConv1d, MinPlusMaxConv2d, MinPlusMaxConv3d
- MinPlusMinConv1d, MinPlusMinConv2d, MinPlusMinConv3d
- MinPlusSumConv1d, MinPlusSumConv2d, MinPlusSumConv3d
- CompoundMinMaxPlusSumConv1d1p, CompoundMinMaxPlusSumConv2d1p, CompoundMinMaxPlusSumConv3d1p
- CompoundMinMaxPlusSumConv1d2p, CompoundMinMaxPlusSumConv2d2p, CompoundMinMaxPlusSumConv3d2p
- ParallelMinMaxPlusSumConv1d1p, ParallelMinMaxPlusSumConv2d1p, ParallelMinMaxPlusSumConv3d1p
- ParallelMinMaxPlusSumConv1d2p, ParallelMinMaxPlusSumConv2d2p, ParallelMinMaxPlusSumConv3d2p
- ConstantCompoundMinMaxPlusSumConv1d, ConstantCompoundMinMaxPlusSumConv2d, ConstantCompoundMinMaxPlusSumConv3d
- ConstantParallelMinMaxPlusSumConv1d, ConstantParallelMinMaxPlusSumConv2d, ConstantParallelMinMaxPlusSumConv3d

## Contributing

Contributions to OpenTCNN are welcome!

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citing OpenTCNN

If you use OpenTCNN in your research, please consider citing our paper or the OpenTCNN project:
- Paper:
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
- OpenTCNN:
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

