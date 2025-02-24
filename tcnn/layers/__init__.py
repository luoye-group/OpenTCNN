# -*- coding: utf-8 -*-
# @Time    : 2023/2/24
# @Author  : LiMingbo
# @E-mail  : limingbo@stu.xmu.edu.cn

from tcnn.layers.base.conv1d import Conv1d
from tcnn.layers.base.conv2d import Conv2d
from tcnn.layers.base.conv3d import Conv3d
from tcnn.layers.base.maxplus_max_conv1d import MaxPlusMaxConv1d
from tcnn.layers.base.maxplus_max_conv2d import MaxPlusMaxConv2d
from tcnn.layers.base.maxplus_max_conv3d import MaxPlusMaxConv3d
from tcnn.layers.base.maxplus_min_conv1d import MaxPlusMinConv1d
from tcnn.layers.base.maxplus_min_conv2d import MaxPlusMinConv2d
from tcnn.layers.base.maxplus_min_conv3d import MaxPlusMinConv3d
from tcnn.layers.base.maxplus_sum_conv1d import MaxPlusSumConv1d
from tcnn.layers.base.maxplus_sum_conv2d import MaxPlusSumConv2d
from tcnn.layers.base.maxplus_sum_conv3d import MaxPlusSumConv3d
from tcnn.layers.base.minplus_max_conv1d import MinPlusMaxConv1d
from tcnn.layers.base.minplus_max_conv2d import MinPlusMaxConv2d
from tcnn.layers.base.minplus_max_conv3d import MinPlusMaxConv3d
from tcnn.layers.base.minplus_min_conv1d import MinPlusMinConv1d
from tcnn.layers.base.minplus_min_conv2d import MinPlusMinConv2d
from tcnn.layers.base.minplus_min_conv3d import MinPlusMinConv3d
from tcnn.layers.base.minplus_sum_conv1d import MinPlusSumConv1d
from tcnn.layers.base.minplus_sum_conv2d import MinPlusSumConv2d
from tcnn.layers.base.minplus_sum_conv3d import MinPlusSumConv3d
from tcnn.layers.compound.compound_min_max_plus_sum_conv1d1p import (
    CompoundMinMaxPlusSumConv1d1p,
)
from tcnn.layers.compound.compound_min_max_plus_sum_conv1d2p import (
    CompoundMinMaxPlusSumConv1d2p,
)
from tcnn.layers.compound.compound_min_max_plus_sum_conv2d1p import (
    CompoundMinMaxPlusSumConv2d1p,
)
from tcnn.layers.compound.compound_min_max_plus_sum_conv2d2p import (
    CompoundMinMaxPlusSumConv2d2p,
)
from tcnn.layers.compound.compound_min_max_plus_sum_conv3d1p import (
    CompoundMinMaxPlusSumConv3d1p,
)
from tcnn.layers.compound.compound_min_max_plus_sum_conv3d2p import (
    CompoundMinMaxPlusSumConv3d2p,
)
from tcnn.layers.parallel.parallel_min_max_plus_sum_conv1d1p import (
    ParallelMinMaxPlusSumConv1d1p,
)
from tcnn.layers.parallel.parallel_min_max_plus_sum_conv1d2p import (
    ParallelMinMaxPlusSumConv1d2p,
)
from tcnn.layers.parallel.parallel_min_max_plus_sum_conv2d1p import (
    ParallelMinMaxPlusSumConv2d1p,
)
from tcnn.layers.parallel.parallel_min_max_plus_sum_conv2d2p import (
    ParallelMinMaxPlusSumConv2d2p,
)
from tcnn.layers.parallel.parallel_min_max_plus_sum_conv3d1p import (
    ParallelMinMaxPlusSumConv3d1p,
)
from tcnn.layers.parallel.parallel_min_max_plus_sum_conv3d2p import (
    ParallelMinMaxPlusSumConv3d2p,
)


from tcnn.layers.compound.constant_compound_min_max_plus_sum_conv1d import (
    ConstantCompoundMinMaxPlusSumConv1d,
)
from tcnn.layers.compound.constant_compound_min_max_plus_sum_conv2d import (
    ConstantCompoundMinMaxPlusSumConv2d,
)
from tcnn.layers.compound.constant_compound_min_max_plus_sum_conv3d import (
    ConstantCompoundMinMaxPlusSumConv3d,
)

from tcnn.layers.parallel.constant_parallel_min_max_plus_sum_conv1d import (
    ConstantParallelMinMaxPlusSumConv1d,
)
from tcnn.layers.parallel.constant_parallel_min_max_plus_sum_conv2d import (
    ConstantParallelMinMaxPlusSumConv2d,
)
from tcnn.layers.parallel.constant_parallel_min_max_plus_sum_conv3d import (
    ConstantParallelMinMaxPlusSumConv3d,
)



__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "MaxPlusMaxConv1d",
    "MaxPlusMaxConv2d",
    "MaxPlusMaxConv3d",
    "MaxPlusMinConv1d",
    "MaxPlusMinConv2d",
    "MaxPlusMinConv3d",
    "MaxPlusSumConv1d",
    "MaxPlusSumConv2d",
    "MaxPlusSumConv3d",
    "MinPlusMaxConv1d",
    "MinPlusMaxConv2d",
    "MinPlusMaxConv3d",
    "MinPlusMinConv1d",
    "MinPlusMinConv2d",
    "MinPlusMinConv3d",
    "MinPlusSumConv1d",
    "MinPlusSumConv2d",
    "MinPlusSumConv3d",
    "CompoundMinMaxPlusSumConv1d1p",
    "CompoundMinMaxPlusSumConv1d2p",
    "CompoundMinMaxPlusSumConv2d1p",
    "CompoundMinMaxPlusSumConv2d2p",
    "CompoundMinMaxPlusSumConv3d1p",
    "CompoundMinMaxPlusSumConv3d2p",
    "ParallelMinMaxPlusSumConv1d1p",
    "ParallelMinMaxPlusSumConv1d2p",
    "ParallelMinMaxPlusSumConv2d1p",
    "ParallelMinMaxPlusSumConv2d2p",
    "ParallelMinMaxPlusSumConv3d1p",
    "ParallelMinMaxPlusSumConv3d2p",
    "ConstantCompoundMinMaxPlusSumConv1d",
    "ConstantCompoundMinMaxPlusSumConv2d",
    "ConstantCompoundMinMaxPlusSumConv3d",
    "ConstantParallelMinMaxPlusSumConv1d",
    "ConstantParallelMinMaxPlusSumConv2d",
    "ConstantParallelMinMaxPlusSumConv3d",
]
