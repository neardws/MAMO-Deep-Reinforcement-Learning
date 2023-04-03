from Environments.environment import make_environment_spec
from acme.agents.tf.d4pg.networks import make_default_networks
from Agents.D4PG.agent import D4PGConfig, D4PG
from Environments.environment_loop import EnvironmentLoopforD4PG
from Utilities.FileOperator import load_obj
from Experiment.environment_file_name import environment_file_without_reward_matrix_information as environment_file_list


def main(_):
    
    for environment_file_name in environment_file_list:

        environment = load_obj(environment_file_name)
        
        spec = make_environment_spec(environment)

        # Create the networks.
        networks = make_default_networks(
            action_spec=spec.actions
        )

        agent_config = D4PGConfig()
        agent = D4PG(
            environment_spec=spec,
            policy_network=networks['policy'],
            critic_network=networks['critic'],
            observation_network=networks['observation']
        )
        
        # Create the environment loop used for training.
        train_loop = EnvironmentLoopforD4PG(environment, agent, label='train_loop')

        train_loop.run(num_episodes=5000)
