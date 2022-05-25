import launchpad as lp
import acme
from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAD3PG.networks import make_default_D3PGNetworks
from Agents.MAD3PG.agent import D3PGConfig, MultiAgentDistributedDDPG
from Agents.MAD3PG.networks import make_default_D3PGNetworks
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
        networks=networks,
        num_actors=10,
    )

    program = agent.build()
    
    (learner_node,) = program.groups['learner']
    learner_node.disable_run()
    
    (evaluator_node,) = program.groups['evaluator']
    evaluator_node.disable_run()
    
    actor_nodes = program.groups['actor']
    for actor_node in actor_nodes:
        actor_node.disable_run()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
    
    # learner: acme.Learner = learner_node.create_handle().dereference()
    # evaluator: EnvironmentLoop = evaluator_node.create_handle().dereference()
    # actors: list = [actor_node.create_handle().dereference() for actor_node in actor_nodes]
    
    # num_episodes = 100
    # for _ in range(num_episodes):
        