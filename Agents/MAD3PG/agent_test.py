"""Tests for the D3PG agent."""
import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

from Agents.MAD3PG.environment_loop import EnvironmentLoop
from Environments import specs
from acme.utils import counting
from Agents.MAD3PG.agent import D3PGConfig, D3PGAgent
from Environments.environment import vehicularNetworkEnv
from Test.environmentConfig_test import vehicularNetworkEnvConfig
from absl.testing import absltest
from Agents.MAD3PG.networks import make_default_D3PGNetworks

class D3PGTest(absltest.TestCase):

    def test_d3pg(self):
        # Create a environment to test with.

        config = vehicularNetworkEnvConfig()
        config.vehicle_list_seeds += [i for i in range(config.vehicle_number)]
        config.view_list_seeds += [i for i in range(config.view_number)]

        env = vehicularNetworkEnv(config)

        spec = specs.make_environment_spec(env)

        # Create the networks.
        networks = make_default_D3PGNetworks(
            vehicle_action_spec=spec.vehicle_actions,
            edge_action_spec=spec.edge_actions,
        )

        config = D3PGConfig(
            batch_size=10, samples_per_insert=2, min_replay_size=10)
        counter = counting.Counter()
        agent = D3PGAgent(
            config=config,
            environment=env,
            networks=networks,
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = EnvironmentLoop(env, agent, counter=counter)
        loop.run(num_episodes=2)


if __name__ == '__main__':
    absltest.main()