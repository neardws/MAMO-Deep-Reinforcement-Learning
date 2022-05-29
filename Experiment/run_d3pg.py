from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAD3PG.networks import make_default_D3PGNetworks
from Agents.MAD3PG.agent import D3PGConfig, D3PGAgent
from Agents.MAD3PG.networks import make_default_D3PGNetworks
from Environments.environmentConfig import vehicularNetworkEnvConfig
from Agents.MAD3PG.environment_loop import EnvironmentLoop


def main(_):
    
    environment_config = vehicularNetworkEnvConfig()
    environment_config.vehicle_list_seeds += [i for i in range(environment_config.vehicle_number)]
    environment_config.view_list_seeds += [i for i in range(environment_config.view_number)]

    environment = vehicularNetworkEnv(environment_config)

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

    train_loop.run(num_episodes=100)

    