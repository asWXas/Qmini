"""Resolve data paths relative to the YAML file that declares them."""

import os
from pathlib import Path

import yaml


def resolve_path(value, config_file):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_file).resolve().parent / path
    return str(path.resolve())


def load_config(config_file):
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    # Explicit opt-in for the original repository-root-relative configs.
    base_dir = Path(config_file).resolve().parent
    path_root = config.pop("path_root", None)
    if path_root is not None:
        base_dir = Path(resolve_path(path_root, config_file))
    for key in ("env_config", "motion_file", "char_file"):
        if key in config:
            config[key] = str((base_dir / Path(config[key]).expanduser()).resolve())
    return config


def save_config(config, output_file):
    """Save paths relative to the snapshot so a whole folder can be relocated."""
    output_file = Path(output_file).resolve()
    config = config.copy()
    for key in ("env_config", "motion_file", "char_file"):
        if key in config:
            try:
                config[key] = Path(os.path.relpath(config[key], output_file.parent)).as_posix()
            except ValueError:
                # Windows cannot form a relative path across different drives.
                config[key] = str(Path(config[key]).resolve())
    with output_file.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
