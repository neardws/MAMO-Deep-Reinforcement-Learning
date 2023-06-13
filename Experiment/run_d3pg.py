from Environments.environment import make_environment_spec
from Agents.MAD3PG.networks import make_default_D3PGNetworks
from Agents.MAD3PG.agent import D3PGConfig, D3PGAgent
from Agents.MAD3PG.networks import make_default_D3PGNetworks
from Environments.environment_loop import EnvironmentLoop
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
        agent = D3PGAgent(
            config=agent_config,
            environment=environment,
            environment_spec=spec,
            networks=networks,
        )
        
        # Create the environment loop used for training.
        train_loop = EnvironmentLoop(environment, agent, label='train_loop')

        train_loop.run(num_episodes=5000)
