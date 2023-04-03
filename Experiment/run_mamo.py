import launchpad as lp
from Environments.environment import make_environment_spec
from Agents.MAMO.networks import make_default_MAMOD3PGNetworks
from Agents.MAMO.agent import D3PGConfig, MAMODistributedDDPG

from Utilities.FileOperator import load_obj
from Experiment.environment_file_name import environment_file_with_reward_matrix_bandwidth as environment_file_list


def main(_):
    
    for environment_file_name in environment_file_list:

        environment = load_obj(environment_file_name)

        spec = make_environment_spec(environment)

        # Create the networks.
        networks = make_default_MAMOD3PGNetworks(
            vehicle_action_spec=spec.vehicle_actions,
            edge_action_spec=spec.edge_actions,

            vehicle_number=environment._config.vehicle_number,
            vehicle_action_number=environment._vehicle_action_size,
            vehicle_observation_size=environment._vehicle_observation_size,
            edge_observation_size=environment._edge_observation_size,
            edge_action_number=environment._edge_action_size,
            # random_action_size=environment._config.random_action_size,
            weights_number=environment._config.weighting_number,
        )

        agent_config = D3PGConfig()

        agent = MAMODistributedDDPG(
            config=agent_config,
            environment_file_name=environment_file_name,
            environment_spec=spec,
            max_actor_steps=300 * 5000,
            networks=networks,
            num_actors=10,
        )

        program = agent.build()
        
        lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        