import launchpad as lp
from Environments.environment import make_environment_spec
from Agents.MAD3PG.networks import make_default_D3PGNetworks
from Agents.MAD3PG.agent import D3PGConfig, MultiAgentDistributedDDPG
from Utilities.FileOperator import load_obj
from Experiment.environment_file_name import environment_file_without_reward_matrix_information as environment_file_list

def main(_):
    for environment_file_name in environment_file_list:

        environment = load_obj(environment_file_name)
        
        spec = make_environment_spec(environment)

        # Create the networks.
        networks = make_default_D3PGNetworks(
            vehicle_action_spec=spec.vehicle_actions,
            edge_action_spec=spec.edge_actions,
        )

        agent_config = D3PGConfig()

        agent = MultiAgentDistributedDDPG(
            config=agent_config,
            environment_file_name=environment_file_name,
            environment_spec=spec,
            max_actor_steps=300 * 5000,
            networks=networks,
            num_actors=10,
        )

        program = agent.build()
        
        lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        