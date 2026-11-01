"""A failed or hung run must end instead of hanging (docs/DECISIONS.md D83).

    python -m pytest tests/test_shutdown.py
"""
import glob
import multiprocessing
import os
import time

import pytest
import yaml

from utils.multiprocessing_utils import stop_process

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("path", sorted(
    glob.glob(os.path.join(REPO, "configs", "**", "base_config.yaml"), recursive=True)
    + glob.glob(os.path.join(REPO, "configs", "live", "*.yaml"))))
def test_every_root_config_states_the_shutdown_timeout(path):
    # slam.py requires it (no silent default): seconds the clean shutdown waits
    # for the backend before stopping it by force.
    with open(path) as f:
        cfg = yaml.full_load(f)
    t = cfg["Results"]["shutdown_timeout_s"]
    assert isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0, path


def test_stop_process_ends_a_process_that_never_returns():
    p = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(1000,))
    p.start()
    t0 = time.monotonic()
    assert stop_process(p) is True
    assert not p.is_alive() and time.monotonic() - t0 < 30
    assert stop_process(p) is False      # already ended
    assert stop_process(None) is False   # never started


def test_parent_process_is_visible_to_the_backend():
    # The backend stops when mp.parent_process() is no longer alive; in the
    # main process there is no parent (the check is skipped).
    torch_mp = pytest.importorskip("torch.multiprocessing")
    assert torch_mp.parent_process() is None
