"""Tests for the random agent."""

import acme
from acme import specs

from Agents.RA.actors import RandomAgent
from acme.testing import fakes

from absl.testing import absltest


class RandomAgentTest(absltest.TestCase):

    def test_ra(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(episode_length=10, bounded=True)
        spec = specs.make_environment_spec(environment)

        # Construct the agent.
        agent = RandomAgent(
            environment_spec=spec,
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(environment, agent)
        loop.run(num_episodes=2)

        # Imports check.


if __name__ == '__main__':

    absltest.main()