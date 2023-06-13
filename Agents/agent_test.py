"""Tests for the D4PG agent."""

import sys
from typing import Dict, Sequence

import acme
from acme import specs
from acme import types
from acme.agents.tf import d4pg
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf

from absl.testing import absltest


def make_networks(
    action_spec: types.NestedSpec,
    policy_layer_sizes: Sequence[int] = (10, 10),
    critic_layer_sizes: Sequence[int] = (10, 10),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Dict[str, snt.Module]:
    """Creates networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)
    policy_layer_sizes = list(policy_layer_sizes) + [num_dimensions]

    policy_network = snt.Sequential(
        [networks.LayerNormMLP(policy_layer_sizes), tf.tanh])
    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            critic_network=networks.LayerNormMLP(
                critic_layer_sizes, activate_final=True)),
        networks.DiscreteValuedHead(vmin, vmax, num_atoms)
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
    }


class D4PGTest(absltest.TestCase):

    def test_d4pg(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(episode_length=10, bounded=True)
        spec = specs.make_environment_spec(environment)

        # Create the networks.
        agent_networks = make_networks(spec.actions)

        # Construct the agent.
        agent = d4pg.D4PG(
            environment_spec=spec,
            policy_network=agent_networks['policy'],
            critic_network=agent_networks['critic'],
            batch_size=10,
            samples_per_insert=2,
            min_replay_size=10,
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(environment, agent)
        loop.run(num_episodes=2)

        # Imports check


if __name__ == '__main__':
    absltest.main()