"""Autonomous linear recurrent neural networks with whole-block reduction."""

__version__ = "0.1.0"

from .metrics import rmse
from .model import LRNN, ReducedLRNN

__all__ = ["LRNN", "ReducedLRNN", "rmse"]
