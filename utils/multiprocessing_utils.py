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


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
