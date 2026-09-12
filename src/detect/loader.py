"""Sequential GPU model loading (section 12.7).

4 GB of VRAM and a budget of 2.5 GB resident means exactly one GPU model is alive at
a time. This module enforces that rather than trusting everyone to remember: taking a
second model while one is held raises instead of quietly OOM-ing halfway through a demo.

    with gpu_model("yolo", lambda: YOLO("yolo11n.pt")) as model:
        results = model(frame)
    # weights freed, cache emptied, next model may load
"""
import contextlib
import logging

log = logging.getLogger(__name__)

_ACTIVE = None  # name of the model currently holding the GPU, or None


def _torch():
    """Imported lazily so the CPU-only modules (rppg, thermal, fusion) never need torch."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def active():
    return _ACTIVE


def free_vram():
    """Empty the allocator cache. No-op without torch or without CUDA."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_gb():
    """Currently reserved VRAM in GB, or None when there is no GPU to ask."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.memory_reserved() / 1e9


@contextlib.contextmanager
def gpu_model(name, load_fn, budget_gb=2.5):
    """Hold one GPU model for the duration of the block, then free it.

    load_fn is a zero-argument callable so nothing is loaded until the guard has
    confirmed the GPU is free.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError(
            f"cannot load {name!r} while {_ACTIVE!r} holds the GPU. "
            f"Models load sequentially on a 4 GB budget; close the other block first."
        )

    _ACTIVE = name
    model = None
    try:
        model = load_fn()
        used = vram_gb()
        if used is not None:
            log.info("%s loaded, %.2f GB reserved", name, used)
            if used > budget_gb:
                log.warning("%s pushed VRAM to %.2f GB, over the %.1f GB budget",
                            name, used, budget_gb)
        yield model
    finally:
        del model
        free_vram()
        _ACTIVE = None
        log.info("%s released", name)
