# Name -> implementation registry of the light-model components
# (CLAUDE.md section 4.2). A model form is chosen by NAME in the config and
# looked up here; adding a form = adding a class with @register, never an
# if/else at a call site.
#
# Each entry declares which keys of its config block it accepts:
#   numeric - numeric parameters, each a ParamSpec block (params.py);
#   options - non-numeric settings (e.g. lambert's normal_source).
# Any other key in the block is an error (typos must not pass silently).
#
# No torch import here: config parsing stays testable without torch.

KINDS = ("falloff", "angular", "cosine", "normals", "ambient")

_REGISTRY = {kind: {} for kind in KINDS}
_PLANNED = {kind: set() for kind in KINDS}


def register(kind, name, numeric=(), options=()):
    if kind not in _REGISTRY:
        raise ValueError(f"unknown component kind {kind!r}")

    def deco(cls):
        if name in _REGISTRY[kind]:
            raise ValueError(f"{kind} '{name}' registered twice")
        _REGISTRY[kind][name] = (cls, tuple(numeric), tuple(options))
        return cls

    return deco


def declare_planned(kind, *names):
    """Names accepted as known but not implemented yet (NotImplementedError)."""
    _PLANNED[kind].update(names)


def lookup(kind, name, where):
    """(class, numeric keys, option keys) registered for kind/name."""
    if name in _REGISTRY[kind]:
        return _REGISTRY[kind][name]
    if name in _PLANNED[kind]:
        raise NotImplementedError(f"{where}.type '{name}' is planned but not implemented yet")
    known = sorted(_REGISTRY[kind]) + sorted(_PLANNED[kind])
    raise ValueError(f"{where}.type must be one of {known}, got {name!r}")


def names(kind):
    return sorted(_REGISTRY[kind]), sorted(_PLANNED[kind])


def build_component(kind, block, where, params_data):
    """Instantiates the component described by a config block
    {type: <name>, <numeric ParamSpec blocks>, <options>}."""
    from light_models.params import resolve_param
    from utils.config_utils import require_key

    if not isinstance(block, dict):
        raise TypeError(f"{where} must be a block with a 'type', got {block!r}")
    name = require_key(block, "type", where)
    cls, numeric, options = lookup(kind, name, where)
    extra = set(block) - {"type"} - set(numeric) - set(options)
    if extra:
        raise ValueError(
            f"{where}: unexpected keys {sorted(extra)} for type '{name}' "
            f"(allowed: {sorted(set(numeric) | set(options))})"
        )
    kwargs = {
        n: resolve_param(require_key(block, n, where), f"{where}.{n}", params_data)
        for n in numeric
    }
    specs = dict(kwargs)
    kwargs.update({o: require_key(block, o, where) for o in options})
    obj = cls(**kwargs)
    # The resolved numeric ParamSpecs, so the shader can find the learnable
    # ones (light_models/learning.py, D60). Convention: a component keeps a
    # numeric parameter's tensor in the attribute of the same name.
    obj._param_specs = specs
    return obj
