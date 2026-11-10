"""Speed helpers of the frontend that do not change any result
(docs/DECISIONS.md D128).

- frame_constant: per-frame memo of tensors that depend only on the observed
  frame (its image, sensor depth, masks), for the frame being tracked.
- FramePrefetcher: reads the next frame from disk on a worker thread while the
  current one is tracked.
- TimeBudget: where the frontend's wall time goes, logged at the end of a run.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from utils.logging_utils import Log


def frame_constant(camera, key, build):
    """build() once per frame while camera.frame_cache is a dict (FrontEnd.tracking
    sets it for the frame it tracks and clears it afterwards); otherwise build()
    on every call, as before. Only for values that depend on the observation
    alone: they must not change between the iterations of one frame."""
    cache = getattr(camera, "frame_cache", None)
    if cache is None:
        return build()
    if key not in cache:
        cache[key] = build()
    return cache[key]


class FramePrefetcher:
    """dataset[idx] for the frame asked, with idx + 1 already being read on a
    worker thread. Frames are read in order; a frame asked again (monocular
    restart) or out of order is read on the spot. Same values as reading it
    directly: the dataset's __getitem__ depends on idx only."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = {}

    def _submit(self, idx):
        if idx < len(self.dataset) and idx not in self.pending:
            self.pending[idx] = self.pool.submit(self.dataset.__getitem__, idx)

    def get(self, idx):
        future = self.pending.pop(idx, None)
        data = future.result() if future is not None else self.dataset[idx]
        for stale in [i for i in self.pending if i < idx]:
            self.pending.pop(stale).cancel()
        self._submit(idx + 1)
        return data

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pending.clear()


class TimeBudget:
    """with budget("name"): ... adds the wall time of the block to "name"."""

    def __init__(self):
        self.start = time.perf_counter()
        self.seconds = {}

    @contextmanager
    def __call__(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t

    def log(self, n_frames):
        total = time.perf_counter() - self.start
        rest = total - sum(self.seconds.values())
        Log(f"frontend time over {n_frames} frames: {total:.0f} s, {1000 * total / max(n_frames, 1):.0f} ms per frame",
            tag="Eval")
        for name, sec in sorted(self.seconds.items(), key=lambda kv: -kv[1]) + [("other", rest)]:
            Log(f"  {name:32s} {sec:8.1f} s  {100 * sec / total:5.1f}%  {1000 * sec / max(n_frames, 1):7.1f} ms per frame",
                tag="Eval")
