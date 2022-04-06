# """Integration test for the distributed agent."""

# from absl.testing import absltest
# import acme
# from Environments import specs
# from Agents.MAD3PG.agent_distributed import MultiAgentDistributedDDPG
# from acme.testing import fakes
# from acme.tf import networks
# from acme.tf import utils as tf2_utils
# import launchpad as lp
# import numpy as np
# import sonnet as snt


# def make_networks(action_spec: specs.BoundedArray):
#     """Simple networks for testing.."""

#     num_dimensions = np.prod(action_spec.shape, dtype=int)

#     policy_network = snt.Sequential([
#         networks.LayerNormMLP([50], activate_final=True),
#         networks.NearZeroInitializedLinear(num_dimensions),
#         networks.TanhToSpec(action_spec)
#     ])
#     # The multiplexer concatenates the (maybe transformed) observations/actions.
#     critic_network = snt.Sequential([
#         networks.CriticMultiplexer(
#             critic_network=networks.LayerNormMLP(
#                 [50], activate_final=True)),
#         networks.DiscreteValuedHead(-1., 1., 10)
#     ])

#     return {
#         'policy': policy_network,
#         'critic': critic_network,
#         'observation': tf2_utils.batch_concat,
#     }


# class DistributedAgentTest(absltest.TestCase):
#     """Simple integration/smoke test for the distributed agent."""

#     def test_control_suite(self):
#         """Tests that the agent can run on the control suite without crashing."""

#         agent = MultiAgentDistributedDDPG(
#             environment_factory=lambda x: fakes.ContinuousEnvironment(bounded=True),
#             network_factory=make_networks,
#             num_actors=2,
#             batch_size=32,
#             min_replay_size=32,
#             max_replay_size=1000,
#         )
#         program = agent.build()

#         (learner_node,) = program.groups['learner']
#         learner_node.disable_run()

#         lp.launch(program, launch_type='test_mt')

#         learner: acme.Learner = learner_node.create_handle().dereference()

#         for _ in range(5):
#             learner.step()


# if __name__ == '__main__':
#     absltest.main()