"""Generic actor implementation, using TensorFlow and Sonnet."""

from typing import Optional, List
from acme import adders
from acme import core
from acme import types
import numpy as np
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from Environments.environment import vehicularNetworkEnv

tfd = tfp.distributions


class FeedForwardActor(core.Actor):
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


    @tf.function
    def get_vehicle_action(
        self, 
        vehicle_observation: types.NestedTensor,
    ) -> types.NestedArray:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        vehicle_batched_observation = tf2_utils.add_batch_dim(vehicle_observation)
        # Compute the policy, conditioned on the observation.
        vehicle_policy = self._vehicle_policy_network(vehicle_batched_observation)
        # Sample from the policy if it is stochastic.
        vehicle_action = vehicle_policy.sample() if isinstance(vehicle_policy, tfd.Distribution) else vehicle_policy
        return vehicle_action

    @tf.function
    def get_edge_action(
        self,
        edge_observation: types.NestedTensor,
    ) -> types.NestedTensor:
        edge_batched_observation = tf2_utils.add_batch_dim(edge_observation)
        edge_policy = self._edge_policy_network(edge_batched_observation)
        edge_action = edge_policy.sample() if isinstance(edge_policy, tfd.Distribution) else edge_policy
        return edge_action

    def _policy(
        self, 
        vehicle_observations: List[types.NestedTensor],
        edge_observation: types.NestedTensor
    ) -> types.NestedTensor:
        action = []
        for i in range(len(vehicle_observations)):
            vehicle_action = self.get_vehicle_action(vehicle_observations[i])
            action.append(vehicle_action)

        edge_action = self.get_edge_action(edge_observation)
        action.append(edge_action)
        actions = tf.concat(action, axis=1)
        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(actions)


    def select_action(self, observation: np.ndarray) -> types.NestedArray:
        # Pass the observation through the policy network.
        vehicle_observations: List[types.NestedTensor] = vehicularNetworkEnv.get_vehicle_observations(
            vehicle_number=self._vehicle_number, 
            information_number=self._information_number, 
            sensed_information_number=self._sensed_information_number, 
            vehicle_observation_size=self._vehicle_observation_size,
            observation=observation)
        edge_observation: types.NestedTensor = vehicularNetworkEnv.get_edge_observation(observation=observation)

        action = self._policy(vehicle_observations, edge_observation)
        # Return a numpy array with squeezed out batch dimension.
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)