"""Shared FOLPS / fkpt pipeline with switchable loop engines."""
from .table import LoopTable, LoopTerms
from .engines import ENGINES, FFTLogEngine, QuadratureEngine
