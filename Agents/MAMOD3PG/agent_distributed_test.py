
"""Integration test for the distributed agent."""
from os import environ
import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

import acme
import launchpad as lp
from absl.testing import absltest
from Environments.environment import make_environment_spec
from Agents.MAMOD3PG.networks import make_default_MAMOD3PGNetworks
from Agents.MAMOD3PG.agent import D3PGConfig, MAMODistributedDDPG

from Utilities.FileOperator import load_obj
from Experiment.environment_file_name import environment_file_with_reward_matrix_bandwidth as environment_file_list

class DistributedAgentTest(absltest.TestCase):
    """Simple integration/smoke test for the distributed agent."""

    def test_control_suite(self):
        """Tests that the agent can run on the control suite without crashing."""

        for environment_file_name in environment_file_list:
            environment = load_obj(environment_file_name)

            spec = make_environment_spec(environment)

            agent_config = D3PGConfig(
                batch_size=32, 
                min_replay_size=32, 
                max_replay_size=1000,
            )

            # Create the networks.
            networks = make_default_MAMOD3PGNetworks(
                vehicle_action_spec=spec.vehicle_actions,
                edge_action_spec=spec.edge_actions,
                
                random_action_size=environment._config.random_action_size,
                vehicle_number=environment._config.vehicle_number,
                vehicle_action_number=environment._vehicle_action_size,
                vehicle_observation_size=environment._vehicle_observation_size,
                edge_observation_size=environment._edge_observation_size,
                edge_action_number=environment._edge_action_size,
                weights_number=environment._config.weighting_number,
            )

            agent = MAMODistributedDDPG(
                config=agent_config,
                environment_file_name=environment_file_name,
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
    
        