from .models.model import IMCBioBatchNet, SeqBioBatchNet
from .api import correct_batch_effects
from .config import Config, ModelConfig, LossConfig, TrainerConfig
from .train import train
from .module import IMCModule, SeqModule

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("biobatchnet")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "IMCBioBatchNet",
    "SeqBioBatchNet",
    "IMCModule",
    "SeqModule",
    "correct_batch_effects",
    "train",
    "Config",
    "ModelConfig",
    "LossConfig",
    "TrainerConfig",
]
