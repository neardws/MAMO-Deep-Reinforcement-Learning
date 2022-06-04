
from typing import Sequence

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils


class DuellingMLP(snt.Module):
    """A Duelling MLP Q-network."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        action_number: int,
        batch_size: int,
    ):
        super().__init__(name='duelling_q_network')

        self._value_mlp = snt.nets.MLP([*hidden_sizes, 1])
        self._advantage_mlp = snt.nets.MLP([*hidden_sizes, 1])
        self._action_number = action_number
        self._batch_size = batch_size

    def __call__(
        self, 
        observation: types.NestedTensor,
        other_action: types.NestedTensor,
        action: types.NestedTensor) -> tf.Tensor:
        """Forward pass of the duelling network.
        Args:
        inputs: 2-D tensor of shape [batch_size, embedding_size].
        Returns:
        q_values: 2-D tensor of action values of shape [batch_size, 1]
        """

        if hasattr(observation, 'dtype') and hasattr(action, 'dtype'):
            if observation.dtype != action.dtype:
                # Observation and action must be the same type for concat to work
                observation = tf.cast(observation, action.dtype)

        # Concat observations and actions, with one batch dimension.
        inputs = tf2_utils.batch_concat([observation, other_action, action])
        
        # Compute value & advantage for duelling.
        value = self._value_mlp(inputs)  # [B, 1]
        
        advantages = self._advantage_mlp(inputs)  # [B, 1]
        
        new_advantages = advantages
        
        for _ in tf.range(self._batch_size):
            random_action = tf.random.uniform(
                shape=[self._batch_size, self._action_number],
                minval=0,
                maxval=1,
                dtype=tf.float64,
            )
            inputs = tf2_utils.batch_concat([observation, other_action, random_action])
            n_advantages = self._advantage_mlp(inputs)
            new_advantages = tf2_utils.batch_concat([new_advantages, n_advantages])
        
        # Advantages have zero mean.
        mean_value = tf.reduce_mean(new_advantages, axis=-1, keepdims=False)
        q_values = value + advantages - mean_value # [B, 1]

        return q_values