"""TCNN TileLang-backed operators.

This package is used inside training/NAS loops where TileLang JIT compilation and
autotuning can be triggered on-demand. To reduce the chance of apparent deadlocks
from heavy multi-threaded compilation/benchmarking, we set conservative defaults
for TileLang autotuner worker counts unless the user explicitly overrides them.
"""

from __future__ import annotations

import os


# Make TileLang autotuning compilation/bench more predictable.
# NOTE: These are *setdefault* so power users can override via environment.
os.environ.setdefault("TILELANG_AUTO_TUNING_CPU_COUNTS", "1")
os.environ.setdefault("TILELANG_AUTO_TUNING_MAX_CPU_COUNT", "1")
os.environ.setdefault("TILELANG_AUTO_TUNING_CPU_UTILITIES", "0.1")

# Apply monkey patch BEFORE importing any TileLang modules
def _monkey_patch_tilelang():
    """Patch TileLang autotuner for thread safety."""
    try:
        import threading
        import tilelang.autotuner.tuner as _tuner
        _orig_rwt = _tuner.run_with_timeout
        def _safe_run_with_timeout(target_fn, timeout, *args, **kwargs):
            if threading.current_thread() is threading.main_thread():
                return _orig_rwt(target_fn, timeout, *args, **kwargs)
            return target_fn(*args, **kwargs)
        _tuner.run_with_timeout = _safe_run_with_timeout
        return True
    except Exception as e:
        # Silently fail - TileLang might not be installed or API changed
        return False

# Apply the patch when package is imported
_patch_success = _monkey_patch_tilelang()

# Expose the channel_last subpackage (if present) for optional channel-last backends
try:
    from . import channel_last as channel_last  # type: ignore
except Exception:
    channel_last = None

from .base.max_plus_sum_conv1d_warp import MaxPlusSumConv1d
from .base.max_plus_sum_conv2d_sharedOpt_warp import MaxPlusSumConv2d
from .base.max_plus_sum_conv3d_warp import MaxPlusSumConv3d
from .base.fan_conv2d_warp import (
    MaxPlusMinConv2d,
    MaxPlusMaxConv2d,
    MinPlusMinConv2d,
    MinPlusMaxConv2d,
)

from .base.min_plus_sum_conv1d_warp import MinPlusSumConv1d
from .base.min_plus_sum_conv2d_sharedOpt_warp import MinPlusSumConv2d
from .base.min_plus_sum_conv3d_warp import MinPlusSumConv3d

from .compound.compound_min_max_plus_sum_conv1d1p_warp import (
    CompoundMinMaxPlusSumConv1d1p,
)
from .compound.compound_min_max_plus_sum_conv1d2p_warp import (
    CompoundMinMaxPlusSumConv1d2p,
)
from .compound.compound_min_max_plus_sum_conv2d_sharedOpt_warp import (
    CompoundMinMaxPlusSumConv2d,
)
from .compound.compound_min_max_plus_sum_conv2d1p_warp import (
    CompoundMinMaxPlusSumConv2d1p,
)
from .compound.compound_min_max_plus_sum_conv2d2p_sharedOpt_warp import (
    CompoundMinMaxPlusSumConv2d2p,
)
from .compound.compound_min_max_plus_sum_conv3d1p_warp import (
    CompoundMinMaxPlusSumConv3d1p,
)
from .compound.compound_min_max_plus_sum_conv3d2p_warp import (
    CompoundMinMaxPlusSumConv3d2p,
)
from .parallel.parallel_min_max_plus_sum_conv1d1p_warp import (
    ParallelMinMaxPlusSumConv1d1p,
)
from .parallel.parallel_min_max_plus_sum_conv1d2p_warp import (
    ParallelMinMaxPlusSumConv1d2p,
)
from .parallel.parallel_min_max_plus_sum_conv2d1p_warp import (
    ParallelMinMaxPlusSumConv2d1p,
)
from .parallel.parallel_min_max_plus_sum_conv2d2p_sharedOpt_warp import (
    ParallelMinMaxPlusSumConv2d2p,
)
from .parallel.parallel_min_max_plus_sum_conv3d1p_warp import (
    ParallelMinMaxPlusSumConv3d1p,
)
from .parallel.parallel_min_max_plus_sum_conv3d2p_warp import (
    ParallelMinMaxPlusSumConv3d2p,
)

# grouped
from .compound.grouped.compound_min_max_plus_sum_conv2d_warp import (
    CompoundMinMaxPlusSumConv2d_Grouped,
)

from .compound.grouped.compound_min_max_plus_sum_conv2d1p_warp import (
    CompoundMinMaxPlusSumConv2d1p_Grouped,
)


from .compound.grouped.compound_min_max_plus_sum_conv2d2p_warp import (
    CompoundMinMaxPlusSumConv2d2p_Grouped,
)

# pointwise
from .compound.pointwise.compound_min_max_plus_sum_conv2d2p_warp import (
    CompoundMinMaxPlusSumConv2d2p_Pointwise,
)

__all__ = [
    "MaxPlusSumConv1d",
    "MaxPlusSumConv2d",
    "MaxPlusSumConv3d",
    "MaxPlusMinConv2d",
    "MaxPlusMaxConv2d",
    "MinPlusMinConv2d",
    "MinPlusMaxConv2d",
    "MinPlusSumConv1d",
    "MinPlusSumConv2d",
    "MinPlusSumConv3d",
    "CompoundMinMaxPlusSumConv1d1p",
    "CompoundMinMaxPlusSumConv1d2p",
    "CompoundMinMaxPlusSumConv2d",
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
    "CompoundMinMaxPlusSumConv2d_Grouped",
    "CompoundMinMaxPlusSumConv2d1p_Grouped",
    "CompoundMinMaxPlusSumConv2d2p_Grouped",
    "CompoundMinMaxPlusSumConv2d2p_Pointwise",
]
