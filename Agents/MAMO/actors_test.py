import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

import time
from Environments.environment_loop import EnvironmentLoop
from absl.testing import absltest
from Agents.MAMOD3PG import actors
from Environments.environment import make_environment_spec
from Agents.MAMOD3PG.networks import make_policy_network

from Utilities.FileOperator import load_obj
from Experiment.environment_file_name import environment_file_with_reward_matrix_bandwidth as environment_file_list

class ActorTest(absltest.TestCase):


    def test_feedforward(self):
        
        for environment_file_name in environment_file_list:
            start_time = time.time()
            environment = load_obj(environment_file_name)
            
            env_spec = make_environment_spec(environment)

            vehicle_policy_network = make_policy_network(env_spec.vehicle_actions)
            edge_policy_network = make_policy_network(env_spec.edge_actions)

            actor = actors.FeedForwardActor(
                vehicle_policy_network=vehicle_policy_network,
                edge_policy_network=edge_policy_network,
                
                vehicle_number=environment._config.vehicle_number,
                information_number=environment._config.information_number,
                sensed_information_number=environment._config.sensed_information_number,
                vehicle_observation_size=environment._vehicle_observation_size,
                
                vehicle_action_size=environment._vehicle_action_size,
                edge_action_size=environment._edge_action_size,
            )
            loop = EnvironmentLoop(environment, actor)
            loop.run(20)
            end_time = time.time()
            print("time taken: ", end_time - start_time)


if __name__ == '__main__':
    absltest.main()