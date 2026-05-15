"""
Centralized Configuration

Loads settings from configs/default.yaml and provides
typed access to all pipeline parameters.
"""

import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    n_samples: int = 6000
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1


@dataclass
class MACFConfig:
    n_trees: int = 2000
    min_leaf_size: int = 20
    max_depth: int = 8
    honesty_fraction: float = 0.5
    subsample_fraction: float = 0.5


@dataclass
class NuisanceConfig:
    n_folds: int = 5
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05


@dataclass
class EvalTargets:
    pehe: float = 0.08
    auroc: float = 0.78
    coverage: float = 0.90
    e_value: float = 2.0
    inference_p95_ms: float = 10.0


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    macf: MACFConfig = field(default_factory=MACFConfig)
    nuisance: NuisanceConfig = field(default_factory=NuisanceConfig)
    targets: EvalTargets = field(default_factory=EvalTargets)
    output_dir: str = "models"


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from YAML file.
    Falls back to defaults if file not found.
    """
    if config_path is None:
        config_path = os.path.join(
            Path(__file__).parent.parent, "configs", "default.yaml"
        )

    config = PipelineConfig()

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)

        if raw and 'data' in raw:
            for k, v in raw['data'].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        if raw and 'model' in raw:
            if 'macf' in raw['model']:
                for k, v in raw['model']['macf'].items():
                    if hasattr(config.macf, k):
                        setattr(config.macf, k, v)
            if 'nuisance' in raw['model']:
                for k, v in raw['model']['nuisance'].items():
                    if hasattr(config.nuisance, k):
                        setattr(config.nuisance, k, v)

        if raw and 'evaluation' in raw and 'targets' in raw['evaluation']:
            for k, v in raw['evaluation']['targets'].items():
                if hasattr(config.targets, k):
                    setattr(config.targets, k, v)

    return config


# Singleton
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get global config (lazy loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    cfg = load_config()
    print(f"Data samples: {cfg.data.n_samples}")
    print(f"MACF trees: {cfg.macf.n_trees}")
    print(f"Target PEHE: {cfg.targets.pehe}")
    print(f"Target AUROC: {cfg.targets.auroc}")
