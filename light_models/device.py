"""Light parameters live on the CPU (ParamSpec tensors) and are moved to the
GPU at every shading call. A copy from pageable host memory makes the CPU wait
for the GPU, and tracking shades ~60 times per frame (docs/DECISIONS.md D128).
on_device keeps the GPU copy of a tensor until the tensor changes (its version
counter: in-place updates, e.g. the frontend loading the values learned by the
backend). A tensor that needs gradient (a parameter being learned in mapping)
is moved as before, so the gradient still reaches the CPU leaf."""
import weakref

import torch

_COPIES = {}  # id(tensor) -> (weakref to tensor, key, copy)


def on_device(t, device, dtype):
    """t.to(device, dtype), without a new copy while t is unchanged."""
    if t.device == device and t.dtype == dtype:
        return t
    if t.requires_grad and torch.is_grad_enabled():
        return t.to(device, dtype)
    key = (t._version, str(device), dtype)
    entry = _COPIES.get(id(t))
    if entry is not None and entry[0]() is t and entry[1] == key:
        return entry[2]
    copy = t.detach().to(device, dtype)
    if len(_COPIES) > 64:
        # Drop the entries of tensors that no longer exist.
        for k in [k for k, e in _COPIES.items() if e[0]() is None]:
            del _COPIES[k]
    _COPIES[id(t)] = (weakref.ref(t), key, copy)
    return copy
