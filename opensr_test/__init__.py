from opensr_test.config import Config
from opensr_test.dataset import load
from opensr_test.main import Metrics

__all__ = ["Config", "load", "Metrics"]
datasets = ["naip", "spot", "venus", "spain_crops", "spain_urban"]

import importlib.metadata as metadata

__version__ = metadata.version("opensr_test")