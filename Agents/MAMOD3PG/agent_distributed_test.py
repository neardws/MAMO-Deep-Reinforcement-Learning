
"""Integration test for the distributed agent."""
import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

import acme
import launchpad as lp
from absl.testing import absltest
from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAMOD3PG.networks import make_default_D3PGNetworks
from Agents.MAMOD3PG.agent import D3PGConfig, MultiAgentDistributedDDPG
from Test.environmentConfig_test import vehicularNetworkEnvConfig

class DistributedAgentTest(absltest.TestCase):
    """Simple integration/smoke test for the distributed agent."""

    def test_control_suite(self):
        """Tests that the agent can run on the control suite without crashing."""

        env_config = vehicularNetworkEnvConfig()
        env_config.vehicle_list_seeds += [i for i in range(env_config.vehicle_number)]
        env_config.view_list_seeds += [i for i in range(env_config.view_number)]

        env = vehicularNetworkEnv(env_config)

        spec = make_environment_spec(env)

        # Create the networks.
        networks = make_default_D3PGNetworks(
            vehicle_action_spec=spec.vehicle_actions,
            edge_action_spec=spec.edge_actions,
        )

        agent_config = D3PGConfig(
            batch_size=32, 
            min_replay_size=32, 
            max_replay_size=1000,
        )

        agent = MultiAgentDistributedDDPG(
            config=agent_config,
            environment_factory=lambda x: vehicularNetworkEnv(env_config),
            environment_spec=spec,
            networks=networks,
            num_actors=2,
        )

        program = agent.build()

        (learner_node,) = program.groups['learner']
        learner_node.disable_run()

        lp.launch(program, launch_type='test_mt', serialize_py_nodes=False)

        learner: acme.Learner = learner_node.create_handle().dereference()

        for _ in range(5):
            learner.step()


if __name__ == '__main__':
    absltest.main()