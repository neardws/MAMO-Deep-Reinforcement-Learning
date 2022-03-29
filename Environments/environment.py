"""Vehicular Network Environments."""
import dm_env
from dm_env import specs
import numpy as np
from Environments.dataStruct import applicationList, edge, informationList, informationRequirements, timeSlots, vehicleList, viewList

class vehicularNetworkEnv(dm_env.Environment):
    """Vehicular Network Environment built on the dm_env framework."""

    def __init__(
        self, 
        time_slots: timeSlots,
        information_list: informationList,
        vehicle_list: vehicleList,
        edge_node: edge,
        application_list: applicationList,
        view_list: viewList,
        information_requirements: informationRequirements
        ) -> None:
        """Initialize the environment.

        Args:
            time_slots: time slots
            information_list: information list
            vehicle_list: vehicle list
            edge_node: edge node
            application_list: application list
            view_list: view list
            information_requirements: information requirements
        """
        self._time_slots = time_slots
        self._information_list = information_list
        self._vehicle_list = vehicle_list
        self._edge_node = edge_node
        self._application_list = application_list
        self._view_list = view_list
        self._information_requirements = information_requirements

        self._vehicle_number = self._vehicle_list.get_vehicles_number()
        self._max_information_number = self._vehicle_list.get_max_information_number()

        self._vehicle_action_size = self._max_information_number + self._max_information_number + \
            self._max_information_number + 1
            # sensed_information + sensing_frequencies + uploading_priorities + transmission_power
        self._vehicle_action_shpae = (self._vehicle_action_size,)
        self._edge_action_size = self._vehicle_number

        self._action_size = self._vehicle_action_size * self._vehicle_number + self._edge_action_size
        self._action_shpae = (self._action_size,)

        self._vehicle_observation_size = 1 + 1 + 1 + self._max_information_number + self._max_information_number + \
            self._information_number + self._information_number 
            # now_time_slot + vehicle_index + distance + information_canbe_sensed + sensing_cost_of_information + \
            # information_in_edge + information_requried

        self._vehicle_observation_shape = (self._vehicle_observation_size,)

        self._observation_size = 1 + self._vehicle_number + self._max_information_number * 2 * self._vehicle_number + \
            self._information_number + self._information_number
            # now_time_slot + vehicle distances + information_canbe_senseds + sensing_cost_of_informations +  \
            # information_in_edge + information_requried

        self._observation_shape = (self._observation_size,)

        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        Returns the first `TimeStep` of a new episode.
        """
        # TODO: implement
        # 
        self._reset_next_step = False
        return dm_env.restart(self._observation())


    def step(self, action: np.array ) -> dm_env.TimeStep:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """
        # TODO: implement
        if self._reset_next_step:
            return self.reset()
        # do actions
        
        # check for termination
        if self.done:
            self._reset_next_step = True
            return dm_env.termination(observation=self._observation(), reward=self.reward)
        return dm_env.transition(observation=self._observation(), reward=self.reward)

    """Define the observation spaces of vehicle."""
    def vehicle_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=self._vehicle_observation_shape,
            dtype=np.float,
            minimum=np.zeros(self._vehicle_observation_shape),
            maximum=np.ones(self._vehicle_observation_shape)
        )

    """Define the action spaces of vehicle."""
    def vehicle_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=self.vehicle_action_shpae,
            dtype=np.float,
            minimum=np.zeros(self.vehicle_action_shpae),
            maximum=np.ones(self.vehicle_action_shpae)
        )

    """Define the gloabl observation spaces."""
    def observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=self._observation_shape,
            dtype=np.float,
            minimum=np.zeros(self._observation_shape),
            maximum=np.ones(self._observation_shape)
        )
    
    """Define the gloabl action spaces."""
    def action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=self._action_shpae,
            dtype=np.float,
            minimum=np.zeros(self._action_shpae),
            maximum=np.ones(self._action_shpae)
        )
    
    def _observation(self) -> np.ndarray:
        """Return the observation of the environment."""
        # TODO: implement
        return self.state
    

    def _vehicle_observation(self) -> np.ndarray:
        """Return the observation of the environment at each vehicle."""
        pass