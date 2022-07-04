"""Tests for the D3PG agent."""
import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

from Agents.MAMOD3PG.environment_loop import EnvironmentLoop
from acme.utils import counting
from Agents.MAMOD3PG.agent import D3PGConfig, MOD3PGAgent
from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Environments.environmentConfig import vehicularNetworkEnvConfig
# from Test.environmentConfig_test import vehicularNetworkEnvConfig
from absl.testing import absltest
from Agents.MAMOD3PG.networks import make_default_D3PGNetworks

class D3PGTest(absltest.TestCase):

    def test_d3pg(self):
        # Create a environment to test with.

        env_config = vehicularNetworkEnvConfig()
        env_config.vehicle_list_seeds += [i for i in range(env_config.vehicle_number)]
        env_config.view_list_seeds += [i for i in range(env_config.view_number)]

        env = vehicularNetworkEnv(env_config, is_reward_matrix=True)

        spec = make_environment_spec(env)

        # Create the networks.
        networks = make_default_D3PGNetworks(
            vehicle_action_spec=spec.vehicle_actions,
            edge_action_spec=spec.edge_actions,
        )

        agent_config = D3PGConfig(
            batch_size=10, samples_per_insert=2, min_replay_size=10)
        counter = counting.Counter()
        agent = MOD3PGAgent(
            config=agent_config,
            environment=env,
            environment_spec=spec,
            networks=networks,
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = EnvironmentLoop(env, agent, counter=counter)
        loop.run(num_episodes=2)


if __name__ == '__main__':
    absltest.main()