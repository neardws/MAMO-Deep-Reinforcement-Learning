from Environments.environment_loop import EnvironmentLoop
from Environments.environment import make_environment_spec
from Agents.RA.actors import RandomActor
from Utilities.FileOperator import load_obj
from Experiment.environment_file_name import environment_file_with_reward_matrix_bandwidth as environment_file_list

def main(_):
    for environment_file_name in environment_file_list:
        environment = load_obj(environment_file_name)
        num_episodes = 100
        spec = make_environment_spec(environment)

        # Create the agent.
        agent = RandomActor(
            spec=spec,
        )

        # Create the environment loop.
        loop = EnvironmentLoop(environment, agent)

        # Run the environment loop.
        loop.run(num_episodes=num_episodes)