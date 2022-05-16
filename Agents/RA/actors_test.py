import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")

from acme import environment_loop
from acme import specs
from Agents.RA.actors import RandomActor
from acme.testing import fakes
import dm_env
import numpy as np
from absl.testing import absltest

def _make_fake_env() -> dm_env.Environment:
    env_spec = specs.EnvironmentSpec(
        observations=specs.Array(shape=(10, 5), dtype=np.float32),
        actions=specs.DiscreteArray(num_values=3),
        rewards=specs.Array(shape=(), dtype=np.float32),
        discounts=specs.BoundedArray(
            shape=(), dtype=np.float32, minimum=0., maximum=1.),
    )
    return fakes.Environment(env_spec, episode_length=10)

class ActorTest(absltest.TestCase):

    def test_random(self):
        environment = _make_fake_env()
        env_spec = specs.make_environment_spec(environment)

        actor = RandomActor(env_spec)
        loop = environment_loop.EnvironmentLoop(environment, actor)
        loop.run(20)


if __name__ == '__main__':
    absltest.main()