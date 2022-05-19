"""Interface for adders which transmit data to a replay buffer."""

import abc

from acme import types
import Environments


class Adder(abc.ABC):
    """The Adder interface.

    An adder packs together data to send to the replay buffer, and potentially
    performs some reduction/transformation to this data in the process.

    All adders will use this API. Below is an illustrative example of how they
    are intended to be used in a typical RL run-loop. We assume that the
    environment conforms to the dm_env environment API.

    ```python
    # Reset the environment and add the first observation.
    timestep = env.reset()
    adder.add_first(timestep.observation)

    while not timestep.last():
        # Generate an action from the policy and step the environment.
        action = my_policy(timestep)
        timestep = env.step(action)

        # Add the action and the resulting timestep.
        adder.add(action, next_timestep=timestep)
    ```

    Note that for all adders, the `add()` method expects an action taken and the
    *resulting* timestep observed after taking this action. Note that this
    timestep is named `next_timestep` precisely to emphasize this point.
    """

    @abc.abstractmethod
    def add_first(self, timestep: Environments.TimeStep):
        """Defines the interface for an adder's `add_first` method.

        We expect this to be called at the beginning of each episode and it will
        start a trajectory to be added to replay with an initial observation.

        Args:
        timestep: a dm_env TimeStep corresponding to the first step.
        """

    @abc.abstractmethod
    def add(
        self,
        action: types.NestedArray,
        next_timestep: Environments.TimeStep,
        extras: types.NestedArray = (),
    ):
        """Defines the adder `add` interface.

        Args:
        action: A possibly nested structure corresponding to a_t.
        next_timestep: A dm_env Timestep object corresponding to the resulting
            data obtained by taking the given action.
        extras: A possibly nested structure of extra data to add to replay.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the adder's buffer."""

# Placeholder for an Adder alias

