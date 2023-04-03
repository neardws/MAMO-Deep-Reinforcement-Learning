from Environments.environment_loop import EnvironmentLoop
from Agents.MAMO.actors import FeedForwardActor
from Utilities.FileOperator import load_obj
import tensorflow as tf
from Experiment.environment_file_name import environment_pareto as environment_file_list

def main(_):
    for environment_file_name in environment_file_list:
        environment = load_obj(environment_file_name)
        
        num_episodes = 100
        
        base_dir = "/home/neardws/acme/1309d9f8-5423-11ed-b6c0-04d9f5632a58/snapshots/"

        edge_policy_dir = base_dir + "target_edge_policy/"
        vehicle_policy_dir = base_dir + "target_vehicle_policy"
        
        edge_policy_network = tf.saved_model.load(edge_policy_dir)
        vehicle_policy_network = tf.saved_model.load(vehicle_policy_dir)
        
        # Create the agent.
        actor = FeedForwardActor(
            vehicle_policy_network=vehicle_policy_network,
            edge_policy_network=edge_policy_network,
            vehicle_number=environment._config.vehicle_number,
            information_number=environment._config.information_number,
            sensed_information_number=environment._config.sensed_information_number,
            vehicle_observation_size=environment._vehicle_observation_size,
            vehicle_action_size=environment._vehicle_action_size,
            edge_action_size=environment._edge_action_size,
        )

        # Create the environment loop.
        loop = EnvironmentLoop(environment, actor)

        # Run the environment loop.
        loop.run(num_episodes=num_episodes)