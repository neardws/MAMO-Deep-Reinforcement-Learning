import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

from Environments.environment import vehicularNetworkEnv
from Environments.environmentConfig import vehicularNetworkEnvConfig
from Utilities.FileOperator import save_obj, init_file_name

def make_default_environment(
    required_information_number: int = 10,
    transmission_power: float = 100.0,  # mW
    bandwidth: float = 2.0,  # MHz
) -> None:
    environment_config = vehicularNetworkEnvConfig(
        transmission_power=transmission_power,
        bandwidth=bandwidth,
        required_information_number=required_information_number,
    )
    environment_config.vehicle_list_seeds += [i for i in range(environment_config.vehicle_number)]
    environment_config.view_list_seeds += [i for i in range(environment_config.view_number)]

    file_name = init_file_name()
    environment = vehicularNetworkEnv(environment_config, is_reward_matrix=True)
    save_obj(environment, file_name["init_environment_with_reward_matrix_name"])
    environment = vehicularNetworkEnv(environment_config, is_reward_matrix=False)
    save_obj(environment, file_name["init_environment_without_reward_matrix_name"])
    
if __name__ == "__main__":
    # make_default_environment(bandwidth=1)
    # make_default_environment(bandwidth=1.5)
    # make_default_environment(bandwidth=2)
    # make_default_environment(bandwidth=2.5)
    # make_default_environment(bandwidth=3)

    # make_default_environment(transmission_power=50)
    # make_default_environment(transmission_power=75)
    # make_default_environment(transmission_power=125)
    # make_default_environment(transmission_power=150)
    
    # make_default_environment(required_information_number=6)
    make_default_environment(required_information_number=8)
    make_default_environment(required_information_number=12)
    make_default_environment(required_information_number=14)