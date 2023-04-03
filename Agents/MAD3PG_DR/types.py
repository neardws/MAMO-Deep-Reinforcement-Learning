
"""Common types used throughout Acme."""

from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union
from acme import specs

# Define types for nested arrays and tensors.
# TODO(b/144758674): Replace these with recursive type definitions.
NestedArray = Any
NestedTensor = Any

# pytype: disable=not-supported-yet
NestedSpec = Union[
    specs.Array,
    Iterable['NestedSpec'],
    Mapping[Any, 'NestedSpec'],
]
# pytype: enable=not-supported-yet

# TODO(b/144763593): Replace all instances of nest with the tensor/array types.
Nest = Union[NestedArray, NestedTensor, NestedSpec]

TensorTransformation = Callable[[NestedTensor], NestedTensor]
TensorValuedCallable = Callable[..., NestedTensor]


class Batches(int):
    """Helper class for specification of quantities in units of batches.

    Example usage:

        # Configure the batch size and replay size in units of batches.
        config.batch_size = 32
        config.replay_size = Batches(4)

        # ...

        # Convert the replay size at runtime.
        if isinstance(config.replay_size, Batches):
            config.replay_size = config.replay_size * config.batch_size  # int: 128

    """


class Transition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    vehicle_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    vehicle_next_observation: NestedArray
    extras: NestedArray = ()
