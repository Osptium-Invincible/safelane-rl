"""Tiny YAML config system: ``configs/env/*.yaml`` + ``configs/algo/*.yaml`` + CLI overrides.

CLI overrides use dotted keys, e.g. ``algo.hyperparams.gamma=0.95 env.config.vehicles_density=2``.
Values are parsed as YAML, so numbers, booleans and lists work (``seeds=[0,1,2]``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_named(kind: str, name: str) -> dict:
    """Load ``configs/<kind>/<name>.yaml``; supports ``base: other_name`` inheritance."""
    cfg = load_yaml(CONFIG_DIR / kind / f"{name}.yaml")
    own_name = cfg.pop("name", None)
    if "base" in cfg:
        parent = load_named(kind, cfg.pop("base"))
        cfg = deep_update(parent, cfg)
    cfg["name"] = own_name or name
    return cfg


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse_overrides(args: list[str]) -> dict:
    out: dict = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Override must look like key=value, got {arg!r}")
        key, raw = arg.split("=", 1)
        set_dotted(out, key, yaml.safe_load(raw))
    return out


def split_overrides(args: list[str]) -> tuple[dict, dict, dict]:
    """Split ``key=value`` args into (top-level, ``algo.*`` overrides, ``env.*`` overrides)."""
    top: dict = {}
    algo_over: dict = {}
    env_over: dict = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Override must look like key=value, got {arg!r}")
        key, raw = arg.split("=", 1)
        value = yaml.safe_load(raw)
        if key.startswith("algo."):
            set_dotted(algo_over, key[len("algo."):], value)
        elif key.startswith("env."):
            set_dotted(env_over, key[len("env."):], value)
        else:
            set_dotted(top, key, value)
    return top, algo_over, env_over


def build_env_spec(env_cfg: dict, algo_cfg: dict) -> dict:
    """Scenario defines the road/traffic, the agent defines observation/action/wrappers."""
    spec = copy.deepcopy(env_cfg)
    spec.pop("name", None)
    config = spec.setdefault("config", {})
    for key in ("observation", "action"):
        if key in algo_cfg:
            config[key] = copy.deepcopy(algo_cfg[key])
    for key in ("shield", "reward_shaping", "safety"):
        if algo_cfg.get(key) is not None:
            spec[key] = deep_update(spec.get(key) or {}, algo_cfg[key])
    return spec
