"""Objects which specify the input/output spaces of an environment.

This module exposes the same spec classes as `dm_env` as well as providing an
additional `EnvironmentSpec` class which collects all of the specs for a given
environment. An `EnvironmentSpec` instance can be created directly or by using
the `make_environment_spec` helper given a `dm_env.Environment` instance.
"""

from typing import NamedTuple
from acme.types import NestedSpec
from dm_env import specs
from Environments.environment import vehicularNetworkEnv

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class EnvironmentSpec(NamedTuple):
  """Full specification of the domains used by a given environment."""
  observations: NestedSpec
  vehicle_observations: NestedSpec
  vehicle_all_observations: NestedSpec
  edge_observations: NestedSpec
  actions: NestedSpec
  vehicle_actions: NestedSpec
  edge_actions: NestedSpec
  rewards: NestedSpec
  critic_vehicle_actions: NestedSpec
  critic_edge_actions: NestedSpec
  discounts: NestedSpec


def make_environment_spec(environment: vehicularNetworkEnv) -> EnvironmentSpec:
  """Returns an `EnvironmentSpec` describing values used by an environment."""
  return EnvironmentSpec(
      observations=environment.observation_spec(),
      vehicle_observations=environment.vehicle_observation_spec(),
      vehicle_all_observations=environment.vehicle_all_observation_spec(),
      edge_observations=environment.edge_observation_spec(),
      actions=environment.action_spec(),
      vehicle_actions=environment.vehicle_action_spec(),
      edge_actions=environment.edge_action_spec(),
      rewards=environment.reward_spec(),
      critic_vehicle_actions=environment.vehicle_critic_network_action_spec(),
      critic_edge_actions=environment.edge_critic_network_action_spec(),
      discounts=environment.discount_spec())