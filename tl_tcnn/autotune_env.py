
from __future__ import annotations

import os


# Conservative TileLang autotuner defaults (can be overridden from the shell).
# These are intentionally set here because many kernel modules import this helper
# directly without going through `tl_tcnn.__init__`.
os.environ.setdefault("TILELANG_AUTO_TUNING_CPU_COUNTS", "1")
os.environ.setdefault("TILELANG_AUTO_TUNING_MAX_CPU_COUNT", "1")
os.environ.setdefault("TILELANG_AUTO_TUNING_CPU_UTILITIES", "0.1")


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _get_autotune_env():
    """Get autotune budget knobs.

    TileLang kernels in this repo are often compiled and benchmarked on-demand during
    the first forward/backward. Aggressive autotuning can look like a deadlock when
    training end-to-end models.

    Environment overrides:
    - `TCNN_AUTOTUNE_FAST` (0/1)
    - `TCNN_AUTOTUNE_WARMUP` (int)
    - `TCNN_AUTOTUNE_REP` (int)
    - `TCNN_AUTOTUNE_MAX_CONFIGS` (int, 0 = unlimited)
    """

    # Conservative defaults to keep training/NAS responsive.
    fast = _env_flag("TCNN_AUTOTUNE_FAST", True)
    warmup = _env_int("TCNN_AUTOTUNE_WARMUP", 1 if fast else 3)
    rep = _env_int("TCNN_AUTOTUNE_REP", 5 if fast else 20)
    max_configs = _env_int("TCNN_AUTOTUNE_MAX_CONFIGS", 16 if fast else 0)

    warmup = max(0, warmup)
    rep = max(1, rep)
    max_configs = max(0, max_configs)

    return fast, warmup, rep, max_configs
