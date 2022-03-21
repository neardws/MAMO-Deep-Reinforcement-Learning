"""Vehicular Network Environments."""
import dm_env
from dm_env import specs
import numpy as np

class vehicularNetworkEnv(dm_env.Environment):
    """Vehicular Network Environment built on the dm_env framework."""

    def __init__(self, information_set):
        """Initialize the environment.

        Args:
            env_params: Environment parameters.
        """
        self._information_set = env_arams['information_set']

        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        Returns the first `TimeStep` of a new episode.
        """
        # TODO implement
        # 
        self._reset_next_step = False
        return dm_env.restart(self._observation())


    def step(self, action: Any) -> dm_env.TimeStep:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """
        # TODO implement
        if self._reset_next_step:
            return self.reset()
        # do actions

        # check for termination
        if self.done:
            self._reset_next_step = True
            return dm_env.termination(observation=self._observation(), reward=self.reward)
        return dm_env.transition(observation=self._observation(), reward=self.reward)

    def observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(4,),
            dtype=np.float32,
            minimum=np.array([0, 0, 0, 0]),
            maximum=np.array([1, 1, 1, 1]),
        )
    
    def action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(4,),
            dtype=np.float32,
            minimum=np.array([0, 0, 0, 0]),
            maximum=np.array([1, 1, 1, 1]),
        )
    
    def _observation(self) -> np.ndarray:
        """Return the observation of the environment."""
        # TODO implement
        return self.state
    
