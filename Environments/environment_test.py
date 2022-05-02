from Environments.environment import vehicularNetworkEnv
from Test.environmentConfig_test import vehicularNetworkEnvConfig

config = vehicularNetworkEnvConfig()
config.vehicle_list_seeds += [i for i in range(config.vehicle_number)]
config.view_list_seeds += [i for i in range(config.view_number)]

env = vehicularNetworkEnv(config)

def test_size():
    print(
        "vehicle_action_size: ", env._vehicle_action_size, "\n", 
        "edge_action_size: ", env._edge_action_size, "\n",
        "action_size: ", env._action_size, "\n",
        "vehicle_observation_size: ", env._vehicle_observation_size, "\n",
        "edge_observation_size: ", env._edge_observation_size, "\n",
        "observation_size: ", env._observation_size, "\n",
        "reward_size: ", env._reward_size, "\n", 
        "vehicle_critic_network_action_size: ", env._vehicle_critic_network_action_size, "\n",
        "edge_critic_network_action_size", env._edge_critic_network_action_size, "\n")

