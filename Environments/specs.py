"""Objects which specify the input/output spaces of an environment.

This module exposes the same spec classes as `dm_env` as well as providing an
additional `EnvironmentSpec` class which collects all of the specs for a given
environment. An `EnvironmentSpec` instance can be created directly or by using
the `make_environment_spec` helper given a `dm_env.Environment` instance.
"""

from typing import NamedTuple
from acme.types import NestedSpec
import dm_env
from dm_env import specs

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class EnvironmentSpec(NamedTuple):
  """Full specification of the domains used by a given environment."""
  observations: NestedSpec
  vehicle_observations: NestedSpec
  edge_observations: NestedSpec
  actions: NestedSpec
  vehicle_actions: NestedSpec
  edge_actions: NestedSpec
  rewards: NestedSpec
  discounts: NestedSpec


def make_environment_spec(environment: dm_env.Environment) -> EnvironmentSpec:
  """Returns an `EnvironmentSpec` describing values used by an environment."""
  return EnvironmentSpec(
      observations=environment.observation_spec(),
      vehicle_observations=environment.vehicle_observation_spec(),
      edge_observations=environment.edge_observation_spec(),
      actions=environment.action_spec(),
      vehicle_actions=environment.vehicle_action_spec(),
      edge_actions=environment.edge_action_spec(),
      rewards=environment.reward_spec(),
      discounts=environment.discount_spec())