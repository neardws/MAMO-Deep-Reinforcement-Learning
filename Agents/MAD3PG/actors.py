"""Generic actor implementation, using TensorFlow and Sonnet."""

from typing import Optional
from acme import adders
from Agents.MAD3PG import base
from acme import types
import numpy as np
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class FeedForwardActor(base.Actor):
    """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        vehicle_policy_network: snt.Module,
        edge_policy_network: snt.Module,
        
        vehicle_number: int,
        information_number: int,
        sensed_information_number: int,
        vehicle_observation_size: int,

        vehicle_action_size: int,
        edge_action_size: int,

        adder: Optional[adders.Adder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initializes the actor.

        Args:
            vehicle_policy_network: A module which takes observations and outputs
                actions.
            edge_policy_network: A module which takes observations and outputs
                actions.
            adder: the adder object to which allows to add experiences to a
                dataset/replay buffer.
            variable_client: object which allows to copy weights from the learner copy
                of the policy to the actor copy (in case they are separate).
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._vehicle_policy_network = vehicle_policy_network
        self._edge_policy_network = edge_policy_network
        
        self._vehicle_number = vehicle_number
        self._information_number = information_number
        self._sensed_information_number = sensed_information_number
        self._vehicle_observation_size = vehicle_observation_size
        self._vehicle_action_size = vehicle_action_size
        self._edge_action_size = edge_action_size

    @tf.function(experimental_relax_shapes=True)
    def _policy(
        self, 
        vehicle_observations: types.NestedArray,
        edge_observation: types.NestedArray
    ) -> types.NestedArray:
        action = tf.zeros(shape=(1, self._vehicle_number * self._vehicle_action_size + self._edge_action_size), dtype=tf.float64)

        for i in tf.data.Dataset.range(self._vehicle_number):
            vehicle_observation = vehicle_observations[i * self._vehicle_observation_size: (i+1) * self._vehicle_observation_size]
            # Add a dummy batch dimension and as a side effect convert numpy to TF.
            vehicle_batched_observation = tf2_utils.add_batch_dim(vehicle_observation)
            # Compute the policy, conditioned on the observation.
            vehicle_policy = self._vehicle_policy_network(vehicle_batched_observation)
            # Sample from the policy if it is stochastic.
            vehicle_action = vehicle_policy.sample() if isinstance(vehicle_policy, tfd.Distribution) else vehicle_policy

            action = tf.concat(
                [action[:, :i * self._vehicle_action_size], vehicle_action, action[:, (i+1) * self._vehicle_action_size :]], 
                axis=1)

            action.set_shape([1, self._vehicle_number * self._vehicle_action_size + self._edge_action_size])

        edge_batched_observation = tf2_utils.add_batch_dim(edge_observation)
        edge_policy = self._edge_policy_network(edge_batched_observation)
        edge_action = edge_policy.sample() if isinstance(edge_policy, tfd.Distribution) else edge_policy
        action = tf.concat([action[:, :self._vehicle_number * self._vehicle_action_size], edge_action], axis=1)
        action.set_shape([1, self._vehicle_number * self._vehicle_action_size + self._edge_action_size])
        return action

    def select_action(self, observation: types.NestedArray, vehicle_observations: types.NestedArray) -> types.NestedArray:
        # Pass the observation through the policy network.
        action = self._policy(
            vehicle_observations=tf.convert_to_tensor(vehicle_observations, dtype=tf.float64), 
            edge_observation=tf.convert_to_tensor(observation, dtype=tf.float64))
        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)