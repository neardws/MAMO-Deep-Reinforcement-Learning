
"""Integration test for the distributed agent."""
import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

import acme
import launchpad as lp
from absl.testing import absltest
from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAMOD3PG.networks import make_default_MAMOD3PGNetworks
from Agents.MAMOD3PG.agent import D3PGConfig, MAMODistributedDDPG
from Environments.environmentConfig import vehicularNetworkEnvConfig


class DistributedAgentTest(absltest.TestCase):
    """Simple integration/smoke test for the distributed agent."""

    def test_control_suite(self):
        """Tests that the agent can run on the control suite without crashing."""

        env_config = vehicularNetworkEnvConfig()
        env_config.vehicle_list_seeds += [i for i in range(env_config.vehicle_number)]
        env_config.view_list_seeds += [i for i in range(env_config.view_number)]

        env = vehicularNetworkEnv(env_config)

        spec = make_environment_spec(env)

        agent_config = D3PGConfig(
            batch_size=32, 
            min_replay_size=32, 
            max_replay_size=1000,
        )

        # Create the networks.
        networks = make_default_MAMOD3PGNetworks(
            vehicle_action_spec=spec.vehicle_actions,
            edge_action_spec=spec.edge_actions,

            vehicle_number=env._config.vehicle_number,
            vehicle_action_number=env._vehicle_action_size,
            vehicle_observation_size=env._vehicle_observation_size,
            edge_observation_size=env._edge_observation_size,
            edge_action_number=env._edge_action_size,
            
            weights_number=env._config.weighting_number,
        )

        agent = MAMODistributedDDPG(
            config=agent_config,
            environment_factory=lambda x: vehicularNetworkEnv(env_config, is_reward_matrix=True),
            environment_spec=spec,
            networks=networks,
            num_actors=10,
        )
        
        program = agent.build()

        (learner_node,) = program.groups['learner']
        learner_node.disable_run()

        lp.launch(program, launch_type='test_mt', serialize_py_nodes=False)

        learner: acme.Learner = learner_node.create_handle().dereference()

        for _ in range(5):
            learner.step()

"""
I0706 16:26:12.100667 139610159986432 terminal.py:91] [Evaluator] Episode Length = 300 | Episode Return = [[-2.10311166e+00 -3.11250000e+01 -8.74889961e+00]
 [-2.84008729e+00 -1.56250000e+01 -4.20135198e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [-8.66582608e-02  0.00000000e+00 -6.48738858e-02]
 [-2.72184211e-01 -9.50000000e+00 -4.21850831e+00]
 [-2.30496157e-01  0.00000000e+00 -4.21824424e-02]
 [-7.31307991e-01 -3.12500000e+00 -1.20602745e+00]
 [-1.85233718e+00 -2.00000000e+01 -5.95065648e+00]
 [-1.18325585e+00 -3.50000000e+00 -1.90496356e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 2.11018503e+01  2.06616162e+00  1.43994197e+00]
 [ 1.98512893e+02  2.57125000e+02  1.76073076e+02]] | Evaluator Episodes = 400 | Evaluator Steps = 120000 | Label = Evaluator_Loop | Steps Per Second = 88.480
[reverb/cc/client.cc:165] Sampler and server are owned by the same process (660158) so Table priority_table is accessed directly without gRPC.
[reverb/cc/client.cc:165] Sampler and server are owned by the same process (660158) so Table priority_table is accessed directly without gRPC.
[reverb/cc/client.cc:165] Sampler and server are owned by the same process (660158) so Table priority_table is accessed directly without gRPC.
[reverb/cc/client.cc:165] Sampler and server are owned by the same process (660158) so Table priority_table is accessed directly without gRPC.
I0706 16:26:19.062440 139604858287872 savers.py:156] Saving checkpoint: /home/neardws/acme/6506fc50-fd01-11ec-8061-04d9f5632a58/checkpoints/d3pg_learner
I0706 16:26:24.172745 139604858287872 builder_impl.py:779] Assets written to: /home/neardws/acme/6506fc50-fd01-11ec-8061-04d9f5632a58/snapshots/vehicle_policy/assets
I0706 16:26:25.325628 139604858287872 builder_impl.py:779] Assets written to: /home/neardws/acme/6506fc50-fd01-11ec-8061-04d9f5632a58/snapshots/edge_policy/assets
I0706 16:26:25.346481 139604858287872 terminal.py:91] [Learner] Edge Critic Loss = 1.098 | Edge Policy Loss = 0.000 | Evaluator Episodes = 401 | Evaluator Steps = 120300 | Learner Steps = 1 | Learner Walltime = 0 | Vehicle Critic Loss = 1.098 | Vehicle Policy Loss = 0.267
[       OK ] DistributedAgentTest.test_control_suite
----------------------------------------------------------------------
Ran 1 test in 1691.163s

OK
        
"""
if __name__ == '__main__':
    absltest.main()
    
        