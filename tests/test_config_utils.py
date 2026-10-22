"""utils/config_utils.py: inheritance merge and the `_replace` marker (D54).

    python -m pytest tests/test_config_utils.py
"""
import pytest
import yaml

from utils.config_utils import load_config, update_recursive


def test_merge_adds_and_overwrites_but_keeps_parent_keys():
    parent = {"A": {"type": "x", "p": 1, "q": {"r": 2}}, "B": 3}
    update_recursive(parent, {"A": {"type": "y", "q": {"s": 4}}})
    assert parent == {"A": {"type": "y", "p": 1, "q": {"r": 2, "s": 4}}, "B": 3}


def test_replace_drops_the_parent_block_and_the_marker():
    parent = {"Light": {"angular": {"type": "blender_spot", "half_angle_deg": 1, "blend": 2},
                        "falloff": {"type": "inverse_square"}}}
    update_recursive(parent, {"Light": {"angular": {"_replace": True, "type": "uniform"}}})
    assert parent == {"Light": {"angular": {"type": "uniform"},
                                "falloff": {"type": "inverse_square"}}}


def test_replace_works_on_a_missing_block_and_nested_markers():
    parent = {}
    update_recursive(parent, {"A": {"_replace": True, "b": {"_replace": True, "c": 1}}})
    assert parent == {"A": {"b": {"c": 1}}}


@pytest.mark.parametrize("value", [False, "true", 1, None])
def test_replace_must_be_true(value):
    with pytest.raises(ValueError, match="_replace"):
        update_recursive({"A": {"x": 1}}, {"A": {"_replace": value, "y": 2}})


def test_load_config_applies_replace_through_inheritance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parent.yaml").write_text(yaml.dump({"Light": {"ambient": {"type": "m", "c": 0.1}}}))
    (tmp_path / "child.yaml").write_text(yaml.dump(
        {"inherit_from": "parent.yaml", "Light": {"ambient": {"_replace": True, "type": "none"}}}))
    assert load_config("child.yaml")["Light"]["ambient"] == {"type": "none"}
