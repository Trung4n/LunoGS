"""utils/multiprocessing_utils.drain_queue (D62): empties without blocking."""
import queue

from utils.multiprocessing_utils import FakeQueue, drain_queue


def test_drain_empties_and_counts():
    q = queue.Queue()
    for i in range(3):
        q.put(i)
    assert drain_queue(q) == (3, 0)
    assert q.empty()
    assert drain_queue(q) == (0, 0)  # returns at once on an empty queue


def test_drain_accepts_the_fake_queue():
    assert drain_queue(FakeQueue()) == (0, 0)


class _Unreadable:
    """A queue whose second message fails to unpickle, like a message whose
    sender already freed the shared memory it points to (D64)."""

    def __init__(self):
        self.items = ["ok", FileNotFoundError, "ok"]

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        item = self.items.pop(0)
        if item is FileNotFoundError:
            raise FileNotFoundError(2, "No such file or directory")
        return item


def test_drain_skips_tolerated_unpickling_errors():
    assert drain_queue(_Unreadable(), tolerate=(OSError,)) == (3, 1)


def test_drain_propagates_errors_by_default():
    import pytest

    with pytest.raises(FileNotFoundError):
        drain_queue(_Unreadable())
