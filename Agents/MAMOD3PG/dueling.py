
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
        random_action_size: int,
        observation_size: int,
        other_action_size: int,
    ):
        super().__init__(name='duelling_q_network')

        self._value_mlp = snt.nets.MLP([*hidden_sizes, 1])
        self._advantage_mlp = snt.nets.MLP([*hidden_sizes, 1])
        self._action_number = action_number
        self._random_action_size = random_action_size
        self._observation_size = observation_size
        self._other_action_size = other_action_size

    def __call__(
        self,
        inputs: types.NestedTensor,
    ) -> tf.Tensor:
        """Forward pass of the duelling network.
        Args:
        inputs: 2-D tensor of shape [batch_size, embedding_size].
        Returns:
        q_values: 2-D tensor of action values of shape [batch_size, 1]
        """
        
        # Compute value & advantage for duelling.
        value = self._value_mlp(inputs)  # [B, 1]
        advantages = self._advantage_mlp(inputs)  # [B, 1]
        # new_advantages_list = [[] for _ in range(self._action_number)]
        new_advantages_list = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        new_advantages_list = new_advantages_list.write(new_advantages_list.size(), advantages)
        # new_advantages_list.append(advantages)
        batch_size = tf.shape(inputs)[0]
        
        for _ in tf.range(self._random_action_size):
            random_action = tf.random.uniform(
                shape=[batch_size, self._action_number],
                minval=0,
                maxval=1,
                dtype=tf.float64,
            )
            new_inputs = tf2_utils.batch_concat([inputs[:, : self._observation_size + self._other_action_size], random_action])
            n_advantages = self._advantage_mlp(new_inputs)
            new_advantages_list = new_advantages_list.write(new_advantages_list.size(), n_advantages)
        
        new_advantages = new_advantages_list.stack()
        print("new_advantages: ", new_advantages)
        # Advantages have zero mean.
        mean_value = tf.reduce_mean(new_advantages, axis=0, keepdims=False)
        print("mean_value: ", mean_value)
        q_values = value + advantages - mean_value # [B, 1]
        print("q_values: ", q_values)

        return q_values