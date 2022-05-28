import launchpad as lp
from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAD3PG.networks import make_default_D3PGNetworks
from Agents.MAD3PG.agent import D3PGConfig, MultiAgentDistributedDDPG
from Environments.environmentConfig import vehicularNetworkEnvConfig

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

    agent = MultiAgentDistributedDDPG(
        config=agent_config,
        environment_factory=lambda x: vehicularNetworkEnv(environment_config),
        environment_spec=spec,
        max_actor_steps=1000,
        networks=networks,
        num_actors=10,
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        