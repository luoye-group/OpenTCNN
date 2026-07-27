"""Utils definitions for the TCNN module."""

import collections
import torch
from itertools import repeat
from typing import Any, Dict, List

"""implement from PyTorch: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/utils.py"""
__all__ = ["consume_prefix_in_state_dict_if_present"]


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

dtype_map = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
    torch.qint8: "qint8",
    torch.quint8: "quint8",
    torch.qint32: "qint32",
}

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _list_with_default(out_size: List[int], defaults: List[int]) -> List[int]:
    import torch

    if isinstance(out_size, (int, torch.SymInt)):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(f"Input dimension should be at least {len(out_size) + 1}")
    return [
        v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size) :])
    ]


def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

class DataType:
    def __init__(self, dtype: str):
        self.dtype = dtype
        if dtype == "float16":
            self.bits = 16
        elif dtype == "float32" or dtype == "float":
            self.bits = 32
        elif dtype == "float64" or dtype == "double":
            self.bits = 64
        elif dtype == "int8":
            self.bits = 8
        elif dtype == "int16":
            self.bits = 16
        elif dtype == "int32" or dtype == "int":
            self.bits = 32
        elif dtype == "int64" or dtype == "long":
            self.bits = 64
        elif dtype == "uint8":
            self.bits = 8
        elif dtype == "bool":
            self.bits = 1
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

def _kernel_cache_key(*parts):
        """Create a hashable cache key for compiled kernels.

        NOTE:
        - This function is intentionally variadic because we have 1d/2d/3d kernels
            with different parameter lists.
        - Callers should include any knobs that change codegen behavior, such as
            `dtype`, `int_dtype_str`, and `autotune`.
        """
        return tuple(parts)


def maybe_autotune(configs, warmup, rep, enabled):
    """Wrap a TileLang kernel factory with optional autotune.

    When ``enabled`` is True, this behaves like ``@tilelang.autotune(...)``.
    When False, it bypasses autotune and injects the first config as default
    compile-time meta-parameters.
    """

    def _decorator(fn):
        if enabled:
            import tilelang

            return tilelang.autotune(configs=configs, warmup=warmup, rep=rep)(fn)

        default_cfg = configs[0] if configs else {}

        def _wrapped(*args, **kwargs):
            merged = dict(default_cfg)
            merged.update(kwargs)
            return fn(*args, **merged)

        return _wrapped

    return _decorator

