import yaml


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)

    return cfg
 

# A child config block containing `_replace: true` REPLACES the inherited block
# instead of being merged into it (light extension, docs/DECISIONS.md D54).
# Merging can add or overwrite keys but never remove one, so an ablation that
# switches a light component to a form with fewer parameters (e.g.
# Light.angular: uniform under a blender_spot parent) would keep the parent's
# extra keys, which the light-model builders reject. The marker itself never
# reaches the resolved config.
REPLACE_KEY = "_replace"


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if isinstance(v, dict) and REPLACE_KEY in v:
            if v[REPLACE_KEY] is not True:
                raise ValueError(
                    f"'{REPLACE_KEY}' in config block '{k}' must be true, got {v[REPLACE_KEY]!r}"
                )
            fresh = dict()
            update_recursive(fresh, {kk: vv for kk, vv in v.items() if kk != REPLACE_KEY})
            dict1[k] = fresh
            continue
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def require_key(section, key, where):
    """Returns section[key]; raises if it is missing. `where` is the dotted
    path of `section` in the config ("" for the top level), used only in the
    error message.

    Used by every light-extension option: a missing key must stop the run
    instead of falling back to a default (CLAUDE.md section 2).
    """
    if not isinstance(section, dict) or key not in section:
        name = f"{where}.{key}" if where else key
        raise KeyError(f"Missing required config key '{name}'")
    return section[key]
