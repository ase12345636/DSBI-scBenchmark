"""Configuration management using dataclasses."""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """Data loading configuration."""
    path: Optional[str] = None  # .h5ad path; absolute or relative to project root
    preprocess: Optional[bool] = None  # if None, infer from mode
    batch_key: str = "BATCH"
    cell_type_key: str = "celltype"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    latent_sz: int = 20
    dropout: float = 0.1
    use_bn: bool = False

    bio_encoder_layers: List[int] = field(default_factory=lambda: [512, 2048, 2048])
    batch_encoder_layers: List[int] = field(default_factory=lambda: [512])
    decoder_layers: List[int] = field(default_factory=lambda: [2048, 2048, 512])
    bio_classifier_layers: List[int] = field(default_factory=lambda: [512, 2048, 2048])
    batch_classifier_layers: List[int] = field(default_factory=lambda: [512])


@dataclass
class LossConfig:
    """Loss weights configuration."""
    recon: float = 10.0
    discriminator: float = 0.3
    classifier: float = 1.0
    kl_bio: float = 0.001
    kl_batch: float = 0.1
    ortho: float = 0.01
    kl_size: float = 0.002  # only for GeneVAE/seq


@dataclass
class TrainerConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Scheduler
    scheduler_step: int = 50
    scheduler_gamma: float = 0.1

    # Early stopping
    early_stop: int = 5

    # Logging
    save_dir: str = "./saved/"
    save_period: int = 10
    verbosity: int = 2

    # Evaluation
    sampling_fraction: float = 1.0
    eval_seed: int = 42
    inde_eval_period: int = 1  # independence eval every N epochs (0 to disable)


@dataclass
class Config:
    """Master configuration."""
    mode: str = "imc"  # "imc" or "seq"
    seed: int = 42
    preset: str = ""  # dataset key in presets.yaml (e.g. "damond", "pancreas")

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def load(cls, config: str) -> 'Config':
        """Load config from file path or preset name."""
        path = Path(config)
        if path.exists():
            return cls._from_yaml(path)
        return cls._from_preset(config)

    @classmethod
    def _from_yaml(cls, yaml_path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)

        # Handle _base_ file inheritance
        if '_base_' in yaml_dict:
            base_path = yaml_path.parent / yaml_dict.pop('_base_')
            with open(base_path, 'r') as f:
                base_dict = yaml.safe_load(f)
            yaml_dict = _deep_merge(base_dict, yaml_dict)

        # Handle preset inheritance
        if yaml_dict.get('preset'):
            preset_dict = cls._load_preset_dict(yaml_dict['preset'])
            yaml_dict = _deep_merge(preset_dict, yaml_dict)

        return cls._from_dict(yaml_dict)

    @classmethod
    def _from_preset(cls, dataset: str) -> 'Config':
        """Load configuration from preset name."""
        return cls._from_dict(cls._load_preset_dict(dataset))

    @classmethod
    def _load_preset_dict(cls, dataset: str) -> dict:
        """Load preset as dictionary, inheriting from base.yaml."""
        yaml_dir = Path(__file__).parent / 'yaml'

        # Load base.yaml first
        base_path = yaml_dir / 'base.yaml'
        with open(base_path, 'r') as f:
            base_dict = yaml.safe_load(f)

        # Load presets.yaml
        presets_path = yaml_dir / 'presets.yaml'
        with open(presets_path, 'r') as f:
            presets = yaml.safe_load(f)

        if dataset not in presets:
            raise ValueError(f"Dataset '{dataset}' not found. Available: {list(presets.keys())}")

        preset = presets[dataset]
        preset_dict = {
            'mode': preset['mode'],
            'preset': dataset,
            'model': preset.get('model', {}),
            'loss': preset.get('loss', {}),
            'trainer': preset.get('trainer', {}),
        }

        # Merge: base <- preset (preset overrides base)
        return _deep_merge(base_dict, preset_dict)

    @classmethod
    def _from_dict(cls, d: dict) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            mode=d.get('mode', 'imc'),
            seed=d.get('seed', 42),
            preset=d.get('preset', ''),
            data=DataConfig(**d.get('data', {})),
            model=ModelConfig(**d.get('model', {})),
            loss=LossConfig(**d.get('loss', {})),
            trainer=TrainerConfig(**d.get('trainer', {})),
        )

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'mode': self.mode,
            'seed': self.seed,
            'preset': self.preset,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'loss': self.loss.__dict__,
            'trainer': self.trainer.__dict__,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
