import acme
from Environments.environment import vehicularNetworkEnv, make_environment_spec
from Agents.RA.actors import RandomAgent


def run(environment: vehicularNetworkEnv, num_episodes: int):
    """Runs the environment loop for the given environment."""
    spec = make_environment_spec(environment)

    # Create the agent.
    agent = RandomAgent(
        environment_spec=spec,
    )

    # Create the environment loop.
    loop = acme.EnvironmentLoop(environment, agent)

    # Run the environment loop.
    loop.run(num_episodes=num_episodes)