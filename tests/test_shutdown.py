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


def _exit_with(code):
    raise SystemExit(code)


def test_raise_if_dead_stops_a_run_whose_backend_died():
    # D117: the backend died of CUDA out of memory and the frontend waited
    # for its answers until the Kaggle session ended.
    from utils.multiprocessing_utils import raise_if_dead

    raise_if_dead(None, "backend")  # not started: nothing to check
    ctx = multiprocessing.get_context("spawn")
    alive = ctx.Process(target=time.sleep, args=(1000,))
    alive.start()
    try:
        raise_if_dead(alive, "backend")  # still running: no error
    finally:
        stop_process(alive)
    dead = ctx.Process(target=_exit_with, args=(3,))
    dead.start()
    dead.join(timeout=30)
    with pytest.raises(RuntimeError, match="backend process ended unexpectedly \(exit code 3\)"):
        raise_if_dead(dead, "backend")


def test_the_frontend_checks_the_backend_in_its_loop():
    # The check must sit in the main loop of FrontEnd.run, before any wait.
    import inspect

    from utils.slam_frontend import FrontEnd

    source = inspect.getsource(FrontEnd.run)
    loop = source.index("while True:")
    assert source.index('raise_if_dead(self.backend_process, "backend")') > loop
    assert source.index('raise_if_dead(self.backend_process, "backend")') < source.index("time.sleep", loop)
