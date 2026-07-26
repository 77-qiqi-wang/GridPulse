# -*- coding: utf-8 -*-
from pathlib import Path
from types import SimpleNamespace

import yaml


class Config(SimpleNamespace):
    def ensure_dirs(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def load_config(market):
    config_path = Path(__file__).with_name(f"config_{market.lower()}.yaml")
    if not config_path.is_file():
        raise ValueError(f"Unknown market configuration: {market}")
    with config_path.open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f)
    return Config(**values)
