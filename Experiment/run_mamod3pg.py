import launchpad as lp
from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAMOD3PG.networks import make_default_MAMOD3PGNetworks
from Agents.MAMOD3PG.agent import D3PGConfig, MAMODistributedDDPG
from Environments.environmentConfig import vehicularNetworkEnvConfig


def main(_):
    
    environment_config = vehicularNetworkEnvConfig()
    environment_config.vehicle_list_seeds += [i for i in range(environment_config.vehicle_number)]
    environment_config.view_list_seeds += [i for i in range(environment_config.view_number)]

    environment = vehicularNetworkEnv(environment_config, is_reward_matrix=True)

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
        
        weights_number=environment._config.weighting_number,
    )

    agent_config = D3PGConfig()

    agent = MAMODistributedDDPG(
        config=agent_config,
        environment_factory=lambda x: vehicularNetworkEnv(environment_config),
        environment_spec=spec,
        max_actor_steps=300 * 5000,
        networks=networks,
        num_actors=10,
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        