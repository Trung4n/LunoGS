import copy
import queue

import torch


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        # queue.Empty, the exception of multiprocessing queues (original
        # MonoGS raised mp.queues.Empty, which torch.multiprocessing does not
        # have: AttributeError; found with drain_queue, D62).
        raise queue.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def drain_queue(q, tolerate=()):
    """Removes every message currently in `q` WITHOUT blocking; returns
    (messages removed, messages that could not be read). Unlike
    `while not q.empty(): q.get()`, it cannot hang when another process takes
    the last message between empty() and get() (D62).

    tolerate: exception types raised while UNPICKLING a message that are
    counted and skipped instead of propagated. Only for messages being thrown
    away at shutdown: a message whose sender has already exited can reference
    shared memory that no longer exists (FileNotFoundError, D64)."""
    n = bad = 0
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return n, bad
        except tolerate:
            bad += 1
        n += 1


# Seconds a process gets to end after terminate() before kill() (D83).
TERMINATE_GRACE_S = 10


def raise_if_dead(p, name):
    """Raises if process `p` was started and has ended. The frontend waits for
    the backend's answers in a loop; if the backend dies (e.g. CUDA out of
    memory), nothing would ever answer and the run would hang until the
    session limit (D117). None = not started yet, nothing to check."""
    if p is not None and p.exitcode is not None:
        raise RuntimeError(f"{name} process ended unexpectedly (exit code {p.exitcode})")


def stop_process(p):
    """Ends process `p` if it is still alive: terminate (SIGTERM), then kill
    (SIGKILL) if it ignores that. Returns True if it had to be stopped."""
    if p is None or not p.is_alive():
        return False
    p.terminate()
    p.join(timeout=TERMINATE_GRACE_S)
    if p.is_alive():
        p.kill()
        p.join(timeout=TERMINATE_GRACE_S)
    return True


def move_obj(obj, device):
    """The tensors (and dicts of tensors) of a clone_obj snapshot moved to
    `device`, in place; returns obj. For the map sent by a backend running on
    another GPU (slam.py --backend_gpu, docs/DECISIONS.md D128)."""
    for attr, value in list(obj.__dict__.items()):
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.to(device))
        elif isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
            setattr(obj, attr, {k: v.to(device) for k, v in value.items()})
    return obj


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        value = getattr(clone_obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(clone_obj, attr, value.detach().clone())
        elif isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
            # e.g. GaussianModel._extras (per-Gaussian light values, D121):
            # detached like the tensor attributes, so they can be sent to
            # another process.
            setattr(clone_obj, attr, {k: v.detach().clone() for k, v in value.items()})
    return clone_obj
