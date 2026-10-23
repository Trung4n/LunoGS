"""utils/multiprocessing_utils.drain_queue (D62): empties without blocking."""
import queue

from utils.multiprocessing_utils import FakeQueue, drain_queue


def test_drain_empties_and_counts():
    q = queue.Queue()
    for i in range(3):
        q.put(i)
    assert drain_queue(q) == 3
    assert q.empty()
    assert drain_queue(q) == 0  # returns at once on an empty queue


def test_drain_accepts_the_fake_queue():
    assert drain_queue(FakeQueue()) == 0
