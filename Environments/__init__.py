"""A Python interface for reinforcement learning environments."""

from Environments import _environment
from dm_env._metadata import __version__

Environment = _environment.Environment
StepType = _environment.StepType
TimeStep = _environment.TimeStep

# Helper functions for creating TimeStep namedtuples with default settings.
restart = _environment.restart
termination = _environment.termination
transition = _environment.transition
truncation = _environment.truncation