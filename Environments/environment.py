"""Vehicular Network Environments."""
import dm_env
import dataclasses
from dm_env import specs
import numpy as np
from dataStruct import location, timeSlots
from dataStruct import informationList
from dataStruct import applicationList
from dataStruct import edge
from dataStruct import informationRequirements
from dataStruct import vehicleList
from dataStruct import viewList
from dataStruct import vehicleAction
from dataStruct import edgeAction
from dataStruct import information
from utilities import sensingAndQueuing
from utilities import v2iTransmission

@dataclasses.dataclass
class vehicularNetworkEnvConfig:
    """Configuration for the vehicular network environment."""
    
    """Time slot related."""
    time_slot_start: int = 0
    time_slot_end: int = 299
    time_slot_length: int = 1

    """Information list related."""
    information_number: int = 50
    information_list_seed: int = 0
    data_size_low_bound: float = 100      # Bytes
    data_size_high_bound: float = 1 * 1024 * 1024    # Bytes
    data_types_number: int = 20
    update_interval_low_bound: int = 1
    update_interval_up_bound: int = 10

    """Vehicles related."""
    vehicle_number: int = 10
    trajectories_file_name: str = 'CSV/trajectories_20161116_0800_0850.csv'
    sensed_information_number: int = 10  # the maximum number of information, which can be sensed by the vehicle.
    min_sensing_cost: float = 0.1
    max_sensing_cost: float = 1.0
    transmission_power: float = 100.0  # mW
    vehicle_list_seeds: list = []

    """"Edge related."""
    edge_no: int = 0
    edge_location_x: float = 500.0   # meters
    edge_location_y: float = 500.0   # meters
    communication_range: float = 500.0  # meters
    bandwidth: float = 3.0  # MHz

    """View related."""
    view_number: int = 30
    required_information_number: int = 10  # the maximume number of information required by one view.
    view_list_seeds: list = []

    """Application related."""
    application_number: int = 0
    views_per_application: int = 1
    application_list_seed: int = 0

    """Information Requirements related."""
    max_application_number: int = 3
    min_application_number: int = 1
    information_requirements_seed: int = 0

    """Vehicle Trajectories Processor related."""
    trajectories_file_name: str = 'CSV/gps_20161116'
    longitude_min: float = 104.04565967220308
    latitude_min: float = 30.654605745741608
    map_width: float = 1000.0   # meters
    trajectories_time_start: str = '2016-11-16 08:00:00'
    trajectories_time_end: str = '2016-11-16 08:05:00'
    trajectories_out_file_name = 'CSV/trajectories_20161116_0800_0850.csv'

    """V2I Transmission related."""
    white_gaussian_noise: int = -90  # dBm
    mean_channel_fading_gain: float = 2.0 
    second_moment_channel_fadding_gain: float = 0.4
    path_loss_exponent: int = 3
    SNR_target: int = 1 # [1, 10]
    probabiliity_threshold: float = 0.9

    """Age of View related."""
    wight_of_timeliness: float = 0.3
    wight_of_consistency: float = 0.3
    wight_of_redundancy: float = 0.2
    wight_of_cost: float = 0.2


class vehicularNetworkEnv(dm_env.Environment):
    """Vehicular Network Environment built on the dm_env framework."""

    def __init__(self) -> None:
        """Initialize the environment."""
        self._config: vehicularNetworkEnvConfig = vehicularNetworkEnvConfig()

        self._time_slots: timeSlots = timeSlots(
            start=self._config.time_slot_start,
            end=self._config.time_slot_end,
            length=self._config.time_slot_length,
        )

        self._vehicle_list: vehicleList = vehicleList(
            number=self._config.vehicle_number,
            trajectories_file_name=self._config.trajectories_file_name,
            information_number=self._config.information_number,
            sensed_information_number=self._config.sensed_information_number,
            min_sensing_cost=self._config.min_sensing_cost,
            max_sensing_cost=self._config.max_sensing_cost,
            transmission_power=self._config.transmission_power,
            seeds=self._config.vehicle_list_seeds,
        )

        self._edge_node: edge = edge(
            edge_no=self._config.edge_no,
            information_number=self._config.information_number,
            edge_location=location(
                x=self._config.edge_location_x,
                y=self._config.edge_location_y,
            ),
            communication_range=self._config.communication_range,
            bandwidth=self._config.bandwidth,
        )

        self._information_list: informationList = informationList(
            number=self._config.information_number,
            seed=self._config.information_list_seed,
            data_size_low_bound=self._config.data_size_low_bound,
            data_size_high_bound=self._config.data_size_high_bound,
            data_types_number=self._config.data_types_number,
            update_interval_low_bound=self._config.update_interval_low_bound,
            update_interval_up_bound=self._config.update_interval_up_bound,
            vehicle_list=self._vehicle_list,
            edge_node=self._edge_node,
            additive_white_gaussian_noise=self._config.white_gaussian_noise,
            mean_channel_fading_gain=self._config.mean_channel_fading_gain,
            second_moment_channel_fadding_gain=self._config.second_moment_channel_fadding_gain,
            path_loss_exponent=self._config.path_loss_exponent,
        )

        self._application_list: applicationList = applicationList(
            number=self._config.application_number,
            views_per_application=self._config.views_per_application,
            information_list=self._information_list,
            seed=self._config.application_list_seed,
        )

        self._view_list: viewList = viewList(
            number=self._config.view_number,
            information_number=self._config.required_information_number,
            required_information_number=self._config.required_information_number,
            seeds=self._config.view_list_seeds,
        )

        self._information_requirements: informationRequirements = informationRequirements(
            time_slots=self._time_slots,
            max_application_number=self._config.max_application_number,
            min_application_number=self._config.min_application_number,
            application=self._application_list,
            view=self._view_list,
            information=self._information_list,
            seed=self._config.information_requirements_seed,
        )

        self._vehicle_action_size, self._edge_action_size, self._action_size, \
            self._vehicle_observation_size, self._edge_observation_size, self._observation_size, self._reward_size = \
                self._define_size_of_spaces()
    
        """To record the timeliness, consistency, redundancy, and cost of views."""
        self._timeliness_views_history: list = []
        self._consistency_views_history: list = []
        self._redundancy_views_history: list = []
        self._cost_views_history: list = []

        self._reward_history: list = []

        self._reward: np.array = np.zeros(self._reward_size)

        self._information_in_edge: list = []
        for _ in range(self._config.information_number):
            self._information_in_edge.append([])

        self._reset_next_step: bool = True

    def _define_size_of_spaces(self) -> tuple:
        """
        Defined the shape of the action space.
        """
        vehicle_action_size: int = self._config.sensed_information_number + self._config.sensed_information_number + \
            self._config.sensed_information_number + 1
            # sensed_information + sensing_frequencies + uploading_priorities + transmission_power 
        edge_action_size: int = self._config.vehicle_number   # bandwidth_allocation for vehicles
        action_size: int = vehicle_action_size * self._config.vehicle_number + edge_action_size

        """
        Defined the shape of the observation space.
        """
        vehicle_observation_size: int = 1 + 1 + 1 + self._config.sensed_information_number + self._config.sensed_information_number + \
            self._config.information_number + self._config.information_number 
            # now_time_slot + vehicle_index + distance + information_canbe_sensed + sensing_cost_of_information + \
            # information_in_edge + information_requried
        edge_observation_size: int = 1 + self._config.vehicle_number + self._config.sensed_information_number * 2 * self._config.vehicle_number + \
            self._config.information_number + self._config.information_number
            # now_time_slot + vehicle distances + information_canbe_senseds + sensing_cost_of_informations +  \
            # information_in_edge + information_requried
        observation_size: int = edge_observation_size
        
        """
        The reward consists of the difference rewards for vehicles and the global reward.
        reward[-1] is the global reward.
        reward[-2] is the reward for the edge.
        reward[0:vehicle_number] are the vehicle rewards.
        """
        reward_size: int = self._config.vehicle_number + 1 + 1

        return vehicle_action_size, edge_action_size, action_size, \
            vehicle_observation_size, edge_observation_size, observation_size, reward_size


    def reset(self) -> dm_env.TimeStep:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        Returns the first `TimeStep` of a new episode.
        """
        self._time_slots.reset()
        self._information_in_edge.clear()
        for _ in range(self._config.information_number):
            self._information_in_edge.append([])
        self._reset_next_step = False
        return dm_env.restart(self._observation())

    def step(self, action: np.array ) -> dm_env.TimeStep:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """

        if self._reset_next_step:
            return self.reset()
        
        views_required_number, information_type_required_by_views_at_now, vehicle_actions, edge_action = \
            self.transform_action_array_to_actions(action)
        
        """Compute the baseline reward and difference rewards."""
        information_objects_ordered_by_views = self.compute_information_objects(
            views_required_number=views_required_number,
            information_type_required_by_views_at_now=information_type_required_by_views_at_now,
            vehicle_actions=vehicle_actions,
            edge_action=edge_action,
        )
        baseline_reward = self.compute_reward(
            information_objects_ordered_by_views=information_objects_ordered_by_views,
            vehicle_actions=vehicle_actions,
        )
        self._reward[-1] = baseline_reward
        for i in range(self._vehicle_number):
            information_objects_ordered_by_views = self.compute_information_objects(
                views_required_number=views_required_number,
                information_type_required_by_views_at_now=information_type_required_by_views_at_now,
                vehicle_actions=vehicle_actions,
                edge_action=edge_action,
                vehicle_index=i,
            )   
            vehicle_reward = baseline_reward - self.compute_reward(
                information_objects_ordered_by_views=information_objects_ordered_by_views,
                vehicle_actions=vehicle_actions,
                vehicle_index=i,
            )
            self._reward[i] = vehicle_reward
        edge_reward = (baseline_reward - min(self._reward_history)) / (max(self._reward_history) - min(self._reward_history))
        self._reward[-2] = edge_reward

        """Update the information in the edge node."""
        information_objects_ordered_by_views = self.compute_information_objects(
            views_required_number=views_required_number,
            information_type_required_by_views_at_now=information_type_required_by_views_at_now,
            vehicle_actions=vehicle_actions,
            edge_action=edge_action,
        )
        self.update_information_in_edge(
            information_objects_ordered_by_views=information_objects_ordered_by_views,
        )
        
        # check for termination
        if self.done:
            self._reset_next_step = True
            return dm_env.termination(observation=self._observation(), reward=self._reward)
        return dm_env.transition(observation=self._observation(), reward=self._reward)

    def transform_action_array_to_actions(self, action: np.array) -> tuple:
        """Transform the action array to the actions of vehicles and the edge node.
        Args:
            action: the action of the agent.
                which including the action of vehicles and the action of the edge.
                action[0: vehicle_number * vehicle_action_size] are the actions of vehicles.
                action[vehicle_number * vehicle_action_size: ] are the actions of the edge.
        Returns:
            actions: the actions of vehicles and the edge node.
        """ 
        vhielce_action_array = action[0: self._vehicle_number * self._vehicle_action_size]
        edge_action_array = action[self._vehicle_number * self._vehicle_action_size:]
        
        if len(vhielce_action_array) != self._vehicle_number * self._vehicle_action_size or \
            len(edge_action_array) != self._edge_action_size:
            raise ValueError('The length of the action is not correct.')

        vehicle_actions = []
        for i in range(self._vehicle_number):
            vehicle_actions.append(
                vehicleAction.generate_from_np_array(
                    vehicle_no=i,
                    now_time=self._time_slots.now(),
                    vehicle_list=self._vehicle_list,
                    max_information_number=self._max_information_number,
                    network_output=vhielce_action_array[i * self._vehicle_action_size: (i + 1) * self._vehicle_action_size],
                    white_gaussian_noise=self._config.white_gaussian_noise,
                    mean_channel_fading_gain=self._config.mean_channel_fading_gain,
                    second_moment_channel_fadding_gain=self._config.second_moment_channel_fadding_gain,
                    edge_location=self._edge_node.get_edge_location(),
                    path_loss_exponent=self._config.path_loss_exponent,
                    SNR_target=self._config.SNR_target,
                    probabiliity_threshold=self._config.probabiliity_threshold,
                    action_time=self._time_slots.now(),
                )
            )
        
        edge_action = edgeAction.generate_from_np_array(
            now_time=self._time_slots.now(),
            edge_node=self._edge_node,
            action_time=self._time_slots.now(),
            network_output=edge_action_array,
            vehicle_number=self._vehicle_number,
        )
        
        information_required = self._information_requirements.information_required_by_views_at_now(self._time_slots.now())
        views_required_number: int = information_required["views_required_number"]
        information_type_required_by_views_at_now: list = information_required["information_type_required_by_views_at_now"]

        return views_required_number, information_type_required_by_views_at_now, vehicle_actions, edge_action

    def compute_information_objects(
        self, 
        views_required_number: int,
        information_type_required_by_views_at_now: list,
        vehicle_actions: list,
        edge_action: edgeAction,
        vehicle_no: int = -1) -> float:
        """Compute the reward.
        Args:
            views_required_number: the number of views required by applications.
            information_type_required_by_views_at_now: the information type required by the views.
            vehicle_actions: the actions of the vehicle.
            edge_action: the action of the edge node.
            vehicle_no: the index of the vehicle which do nothing, i.e., its action is null.
                the default value is -1, which means no vehicles do nothing.
        Returns:
            reward: the reward of the vehicle.
        """

        information_objects_ordered_by_views = []
        for i in range(views_required_number):
            information_objects_ordered_by_views.append(list())

        for i in range(self._vehicle_number):
            
            if i == vehicle_no:     # the vehicle do nothing
                continue
            
            sensing_and_queuing = sensingAndQueuing(
                vehicle=self._vehicle_list.get_vehicle(i),
                vehicle_action=vehicle_actions[i],
                information_list=self._information_list,
            )
            sensed_information_type = sensing_and_queuing.get_sensed_information_type()
            arrival_intervals = sensing_and_queuing.get_arrival_intervals()
            arrival_moments = sensing_and_queuing.get_arrival_moments()
            updating_moments = sensing_and_queuing.get_updating_moments()
            queuing_times = sensing_and_queuing.get_queuing_times()

            v2i_transmission = v2iTransmission(
                vehicle=self._vehicle_list.get_vehicle(i),
                vehicle_action=vehicle_actions[i],
                edge=self._edge_node,
                edge_action=edge_action,
                arrival_moments=arrival_moments,
                queuing_times=queuing_times,
                white_gaussian_noise=self._config.white_gaussian_noise,
                mean_channel_fadding_gain=self.mean_channel_fadding_gain,
                second_moment_channel_fadding_gain=self.second_moment_channel_fadding_gain,
                path_loss_exponent=self.path_loss_exponent,
            )

            transmission_times = v2i_transmission.get_transmission_times()

            for information_index in range(len(sensed_information_type)):
                infor = information(
                    type=sensed_information_type[information_index],
                    vehicle_no=i,
                    edge_no=self._edge_node.get_edge_no(),
                    updating_moment=updating_moments[information_index],
                    inter_arrival_interval=arrival_intervals[information_index],
                    arrival_moment=arrival_moments[information_index],
                    queuing_time=queuing_times[information_index],
                    transmission_time=transmission_times[information_index],
                    received_moment=arrival_moments[information_index] + queuing_times[information_index] + transmission_times[information_index],
                )

                for view_index in range(len(information_type_required_by_views_at_now)):
                    if infor.get_type() in information_type_required_by_views_at_now[view_index]:
                        information_objects_ordered_by_views[view_index].append(infor)
        
        """If the view is uncomplete, add the missing information from the information in edge to the view."""
        for view_index in range(len(information_type_required_by_views_at_now)):
            for infor_type in information_type_required_by_views_at_now[view_index]:
                infor_type_exist = False
                for infor in information_objects_ordered_by_views[view_index]:
                    if infor.get_type() == infor_type:
                        infor_type_exist = True
                        break
                if not infor_type_exist and len(self._information_in_edge[infor_type]) > 0:
                    infor_in_edge = self._information_in_edge[infor_type][0]
                    information_objects_ordered_by_views[view_index].append(infor_in_edge)

        return information_objects_ordered_by_views

    def update_information_in_edge(self, information_objects_ordered_by_views: list):
        """Update the information in edge.
        Args:
            information_objects_ordered_by_views: the information objects ordered by views.
        """
        for view_index in range(len(information_objects_ordered_by_views)):
            for infor in information_objects_ordered_by_views[view_index]:
                self._information_in_edge[infor.get_type()].append(infor)
                self._information_in_edge[infor.get_type()].sort(key=lambda x: x.get_received_moment(), reverse=True)
        
    def compute_reward(
        self,
        information_objects_ordered_by_views,
        vehicle_actions: list,
        vehicle_no: int = -1) -> float:

        """Compute the timeliness of views"""
        timeliness_views = []
        for information_objects in information_objects_ordered_by_views:
            timeliness_list = []
            for _ in range(self._vehicle_number):
                timeliness_list.append(list())
            for infor in information_objects:
                timeliness_list[infor.get_vehicle_no()].append(
                    infor.get_arrrival_moment() + infor.get_queuing_time() + infor.get_transmission_time() - infor.get_updating_moment()
                )
            timeliness_of_vehicles = []
            for values in timeliness_list:
                timeliness_of_vehicles.append(max(values))
            timeliness = sum(timeliness_of_vehicles)
            timeliness_views.append(timeliness)
            self._timeliness_views_history.append(timeliness)

        """Compute the consistency of views"""
        consistency_views = []
        for information_objects in information_objects_ordered_by_views:
            updating_moments_of_informations = []
            for infor in information_objects:
                updating_moments_of_informations.append(infor.get_updating_moment())
            consistency = max(updating_moments_of_informations) - min(updating_moments_of_informations)
            consistency_views.append(consistency)
            self._consistency_views_history.append(consistency)
                
        """Compute the redundancy of views"""
        redundancy_views = []
        for information_objects in information_objects_ordered_by_views:
            redundancy = 0
            redundancy_list = []
            for _ in range(self._config.information_number):
                redundancy_list.append(list())
            for infor in information_objects:
                redundancy_list[infor.get_type()].append(1)
            for i in range(self._config.information_number):
                if sum(redundancy_list[i]) > 1:
                    redundancy += sum(redundancy_list[i]) - 1
            redundancy_views.append(redundancy)
            self._redundancy_views_history.append(redundancy)

        """Compute the cost of view"""
        cost_views = []
        for information_objects in information_objects_ordered_by_views:
            cost_list = []
            for _ in range(self._vehicle_number):
                cost_list.append(list())
            for infor in information_objects:
                cost_list[infor.get_vehicle_no()].append(
                    self._vehicle_list.get_vehicle(infor.get_vehicle_no()).get_sensing_cost_by_type(infor.get_type()) + \
                        infor.get_transmission_time() * vehicle_actions[infor.get_vehicle_no()].get_transmission_power()
                )
            cost_of_vehicles = []
            for values in cost_list:
                cost_of_vehicles.append(sum(values))
            cost = sum(cost_of_vehicles)
            cost_views.append(cost)
            self._cost_views_history.append(cost)

        """Normalize the timeliness, consistency, redundancy, and cost of views"""
        timeliness_views_normalized = []
        consistency_views_normalized = []
        redundancy_views_normalized = []
        cost_views_normalized = []
        
        for i in range(len(timeliness_views)):
            timeliness_views_normalized.append(
                (timeliness_views[i] - min(self._timeliness_views_history)) / (max(self._timeliness_views_history) - min(self._timeliness_views_history))
            )
            consistency_views_normalized.append(
                (consistency_views[i] - min(self._consistency_views_history)) / (max(self._consistency_views_history) - min(self._consistency_views_history))
            )
            redundancy_views_normalized.append(
                (redundancy_views[i] - min(self._redundancy_views_history)) / (max(self._redundancy_views_history) - min(self._redundancy_views_history))
            )
            cost_views_normalized.append(
                (cost_views[i] - min(self._cost_views_history)) / (max(self._cost_views_history) - min(self._cost_views_history))
            )

        """Compute the age of view."""
        age_of_view = []
        for i in range(len(timeliness_views_normalized)):
            age_of_view.append(
                self._config.wight_of_timeliness * timeliness_views_normalized[i] + \
                self._config.wight_of_consistency * consistency_views_normalized[i] + \
                self._config.wight_of_redundancy * redundancy_views_normalized[i] + \
                self._config.wight_of_cost * cost_views_normalized[i]
            )

        """Compute the reward."""
        reward = float(1.0 - sum(age_of_view) / len(age_of_view))
        reward = 0 if reward < 0 else reward
        reward = 1 if reward > 1 else reward

        if vehicle_no == -1:
            self._reward_history.append(reward)

        return reward

    """Define the observation spaces of vehicle."""
    def vehicle_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(self._vehicle_observation_size,),
            dtype=np.float,
            minimum=np.zeros((self._vehicle_observation_size,)),
            maximum=np.ones((self._vehicle_observation_size,))
        )

    """Define the action spaces of vehicle."""
    def vehicle_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._vehicle_action_size,),
            dtype=np.float,
            minimum=np.zeros((self.vehicle_action_size,)),
            maximum=np.ones((self.vehicle_action_size,))
        )
    
    """Define the observation spaces of edge."""
    def edge_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(self._edge_observation_size,),
            dtype=np.float,
            minimum=np.zeros((self._edge_observation_size,)),
            maximum=np.ones((self._edge_observation_size,))
        )

    def edge_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._edge_action_size,),
            dtype=np.float,
            minimum=np.zeros((self._edge_action_size,)),
            maximum=np.ones((self._edge_action_size,))
        )

    """Define the gloabl observation spaces."""
    def observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(self._observation_size,),
            dtype=np.float,
            minimum=np.zeros((self._observation_size,)),
            maximum=np.ones((self._observation_size,)),
            name='observation'
        )
    
    """Define the gloabl action spaces."""
    def action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._action_size,),
            dtype=np.float,
            minimum=np.zeros((self._action_size,)),
            maximum=np.ones((self._action_size,)),
            name='action'
        )

    def reward_spec(self):
        """Define and return the reward space."""
        return specs.Array(
            shape=(self._reward_size,), 
            dtype=np.float, 
            name='reward'
        )
    
    def _observation(self) -> np.ndarray:
        """Return the observation of the environment."""
        """
        edge_observation_size: int = 1 + self._config.vehicle_number + self._config.sensed_information_number * 2 * self._config.vehicle_number + \
            self._config.information_number + self._config.information_number
            # now_time_slot + vehicle distances + information_canbe_senseds + sensing_cost_of_informations +  \
            # information_in_edge + information_requried
        """
        observation = np.zeros((self._observation_size,))
        index = 0
        # now_time_slot
        observation[index] = float(self._time_slots.now() / self._time_slots.get_number())
        index += 1
        # vehicle distances
        for _ in range(self._config.vehicle_number):
            observation[index] = float(self._vehicle_list.get_vehicle(_).get_distance_between_edge(
                nowTimeSlot=self._time_slots.now(),
                edge_location=self._edge_node.get_edge_location(),
            ) / (self._edge_node.get_communication_range() * np.sqrt(2)))
            index += 1
        # information_canbe_senseds
        for _ in range(self._config.vehicle_number):
            for __ in range(self._config.sensed_information_number):
                observation[index] = float(self._vehicle_list.get_vehicle(_).get_information_canbe_sensed()[__]
                    / self._config.information_number)
                index += 1
        # sensing_cost_of_informations
        for _ in range(self._config.vehicle_number):
            for __ in range(self._config.sensed_information_number):
                observation[index] = float(self._vehicle_list.get_vehicle(_).get_sensing_cost()[__]
                    / self._config.max_sensing_cost)
                index += 1
        # information_in_edge
        for _ in range(self._config.information_number):
            if len(self._information_in_edge[_]) == 0:
                observation[index] = 0
            else:
                observation[index] = float(self._information_in_edge[_][0].get_received_moment() 
                    / self._time_slots.get_number())
            index += 1
        # information_requried
        for _ in range(self._config.information_number):
            observation[index] = float(self._information_requirements.information_required_at_now()[_])
            index += 1

        return observation

    def _vehicle_observation(self, vehicle_index: int) -> np.ndarray:
        """Return the observation of the environment at each vehicle."""
        """
        vehicle_observation_size: int = 1 + 1 + 1 + self._config.sensed_information_number + self._config.sensed_information_number + \
            self._config.information_number + self._config.information_number 
            # now_time_slot + vehicle_index + distance + information_canbe_sensed + sensing_cost_of_information + \
            # information_in_edge + information_requried
        """
        vehicle_observation = np.zeros((self._vehicle_observation_size,))
        index = 0
        # now_time_slot
        vehicle_observation[index] = float(self._time_slots.now() / self._time_slots.get_number())
        index += 1
        # vehicle_index
        vehicle_observation[index] = float(vehicle_index / self._config.vehicle_number)
        index += 1
        # vehicle distances
        vehicle_observation[index] = float(self._vehicle_list.get_vehicle(vehicle_index).get_distance_between_edge(
                nowTimeSlot=self._time_slots.now(),
                edge_location=self._edge_node.get_edge_location(),
            ) / (self._edge_node.get_communication_range() * np.sqrt(2)))
        index += 1
        # information_canbe_senseds
        for __ in range(self._config.sensed_information_number):
            vehicle_observation[index] = float(self._vehicle_list.get_vehicle(vehicle_index).get_information_canbe_sensed()[__]
                / self._config.information_number)
            index += 1
        # sensing_cost_of_informations
        for __ in range(self._config.sensed_information_number):
            vehicle_observation[index] = float(self._vehicle_list.get_vehicle(vehicle_index).get_sensing_cost()[__]
                / self._config.max_sensing_cost)
            index += 1
        # information_in_edge
        for _ in range(self._config.information_number):
            if len(self._information_in_edge[_]) == 0:
                vehicle_observation[index] = 0
            else:
                vehicle_observation[index] = float(self._information_in_edge[_][0].get_received_moment() 
                    / self._time_slots.get_number())
            index += 1
        # information_requried
        for _ in range(self._config.information_number):
            vehicle_observation[index] = float(self._information_requirements.information_required_at_now()[_])
            index += 1

    def _edge_observation(self) -> np.ndarray:
        """Return the observation of the environment at edge."""
        return self._observation()