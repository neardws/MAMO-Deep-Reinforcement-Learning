
"""Vehicular Network Environments."""
from dm_env import specs
from acme.types import NestedSpec
import numpy as np
from Environments.dataStruct import applicationList, edge, edgeAction, informationList, informationPacket, informationRequirements, location, timeSlots, vehicleAction, vehicleList, viewList  
from typing import List, Optional, Tuple, NamedTuple, Dict
from Environments._environment import baseEnvironment, TimeStep, restart, termination, transition
import Environments.environmentConfig as env_config
from Environments.utilities import sensingAndQueuing, v2iTransmission

class vehicularNetworkEnv(baseEnvironment):
    """Vehicular Network Environment built on the dm_env framework."""
    
    def init_reward_history(self, time_slots_number, is_reward_matrix) -> None:
        self.is_reward_matrix = is_reward_matrix
        """Set the reward history."""
        if self.is_reward_matrix:
            self.reward_history = [{
                "aov_max": -100, 
                "aov_min": 100,
                "cost_max": -100,
                "cost_min": 100,
            } for _ in range(time_slots_number)]
        else:
            self.reward_history = [{"max": -100, "min": 100} for _ in range(time_slots_number)]
            
    def append_reward_at_now(
        self, now: int, 
        reward: Optional[float]=None, 
        aov: Optional[float]=None, 
        cost: Optional[float]=None
    ) -> None:
        if self.is_reward_matrix:
            if aov > self.reward_history[now]["aov_max"]:
                self.reward_history[now]["aov_max"] = aov
            if aov < self.reward_history[now]["aov_min"]:
                self.reward_history[now]["aov_min"] = aov
            if cost > self.reward_history[now]["cost_max"]:
                self.reward_history[now]["cost_max"] = cost
            if cost < self.reward_history[now]["cost_min"]:
                self.reward_history[now]["cost_min"] = cost
        else:
            if reward > self.reward_history[now]["max"]:
                self.reward_history[now]["max"] = reward
            if reward < self.reward_history[now]["min"]:
                self.reward_history[now]["min"] = reward

    def get_min_reward_at_now(self, now: int):
        if self.is_reward_matrix:
            return self.reward_history[now]["aov_min"], self.reward_history[now]["cost_min"]
        else:
            return self.reward_history[now]["min"]

    def get_max_reward_at_now(self, now: int):
        if self.is_reward_matrix:
            return self.reward_history[now]["aov_max"], self.reward_history[now]["cost_max"]
        else:
            return self.reward_history[now]["max"]
        
    def __init__(
        self, 
        envConfig: env_config.vehicularNetworkEnvConfig = None,
        is_reward_matrix: bool = True,
    ) -> None:
        """Initialize the environment."""
        self.reward_history: List[Dict[str, float]] = None
        self.is_reward_matrix: bool = is_reward_matrix
        
        if envConfig is None:
            self._config = env_config.vehicularNetworkEnvConfig()
        else:
            self._config = envConfig

        self.init_reward_history(self._config.time_slot_number, is_reward_matrix)

        self._time_slots: timeSlots = timeSlots(
            start=self._config.time_slot_start,
            end=self._config.time_slot_end,
            slot_length=self._config.time_slot_length,
        )
        
        self._channel_fading_gains = self.generate_channel_fading_gain(
            mean_channel_fading_gain=self._config.mean_channel_fading_gain,
            second_moment_channel_fading_gain=self._config.second_moment_channel_fading_gain,
            size=100
        )
        
        self._successful_tansmission_probability: Dict = {}
        
        self._vehicle_list: vehicleList = vehicleList(
            number=self._config.vehicle_number,
            time_slots=self._time_slots,
            trajectories_file_name=self._config.trajectories_out_file_name,
            information_number=self._config.information_number,
            sensed_information_number=self._config.sensed_information_number,
            min_sensing_cost=self._config.min_sensing_cost,
            max_sensing_cost=self._config.max_sensing_cost,
            transmission_power=self._config.transmission_power,
            seeds=self._config.vehicle_list_seeds,
        )

        self._edge_node: edge = edge(
            edge_index=self._config.edge_index,
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
            data_size_up_bound=self._config.data_size_up_bound,
            data_types_number=self._config.data_types_number,
            update_interval_low_bound=self._config.update_interval_low_bound,
            update_interval_up_bound=self._config.update_interval_up_bound,
            vehicle_list=self._vehicle_list,
            edge_node=self._edge_node,
            white_gaussian_noise=self._config.white_gaussian_noise,
            mean_channel_fading_gain=self._config.mean_channel_fading_gain,
            second_moment_channel_fading_gain=self._config.second_moment_channel_fading_gain,
            path_loss_exponent=self._config.path_loss_exponent,
        )

        self._application_list: applicationList = applicationList(
            number=self._config.application_number,
            views_per_application=self._config.views_per_application,
            view_number=self._config.view_number,
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
            application_list=self._application_list,
            view_list=self._view_list,
            information_list=self._information_list,
            seed=self._config.information_requirements_seed,
        )

        self._vehicle_action_size, self._edge_action_size, self._action_size, \
            self._vehicle_observation_size, self._edge_observation_size, self._observation_size, \
            self._reward_size, self._vehicle_critic_network_action_size, self._edge_critic_network_action_size = \
                self._define_size_of_spaces()
                
    
        """To record the timeliness, consistency, redundancy, and cost of views."""
        self._max_timeliness: float = -1
        self._min_timeliness: float = 100000000
        self._max_consistency: float = -1
        self._min_consistency: float = 100000000
        self._max_redundancy: float = -1
        self._min_redundancy: float = 100000000
        self._max_sensing_cost: float = -1
        self._min_sensing_cost: float = 100000000
        self._max_transmission_cost: float = -1
        self._min_transmission_cost: float = 100000000
        
        if self.is_reward_matrix:
            self._reward: np.ndarray = np.zeros(shape=(self._reward_size, self._config.weighting_number))
        else:
            self._reward: np.ndarray = np.zeros(shape=(self._reward_size,))
            
        self._weights = np.zeros(shape=(self._config.weighting_number,))
        
        self._information_in_edge: List[List[informationPacket]] = []

        for information_type in range(self._config.information_number):
            self._information_in_edge.append([
                informationPacket(
                    type=information_type,
                    edge_index=self._config.edge_index,
                    updating_moment=0,
                    inter_arrival_interval=-1,
                    arrival_moment=0,
                    queuing_time=0,
                    transmission_time=0,
                    received_moment=0,
                )
            ])

        self._reset_next_step: bool = True

    def _define_size_of_spaces(self) -> Tuple[int, int, int, int, int, int, int, int, int]:
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
        
        """Defined the shape of the action space in critic network."""
        vehicle_critic_network_action_size: int = self._config.vehicle_number * vehicle_action_size
        edge_critic_network_action_size: int = self._config.vehicle_number * vehicle_action_size + edge_action_size

        return vehicle_action_size, edge_action_size, action_size, \
            vehicle_observation_size, edge_observation_size, observation_size, \
            reward_size, vehicle_critic_network_action_size, edge_critic_network_action_size


    def reset(self) -> TimeStep:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        Returns the first `TimeStep` of a new episode.
        """
        self._time_slots.reset()
        self._information_in_edge.clear()
        for information_type in range(self._config.information_number):
            self._information_in_edge.append([
                informationPacket(
                    type=information_type,
                    edge_index=self._config.edge_index,
                    updating_moment=0,
                    inter_arrival_interval=-1,
                    arrival_moment=0,
                    queuing_time=0,
                    transmission_time=0,
                    received_moment=0,
                )
            ])
        self._channel_fading_gains = self.generate_channel_fading_gain(
            mean_channel_fading_gain=self._config.mean_channel_fading_gain,
            second_moment_channel_fading_gain=self._config.second_moment_channel_fading_gain,
            size=100
        )
        self._successful_tansmission_probability.clear()
        self._reset_next_step = False
        observation = self._observation()
        vehicle_observation = self.get_vehicle_observations(
            vehicle_number=self._config.vehicle_number,
            information_number=self._config.information_number,
            sensed_information_number=self._config.sensed_information_number,
            vehicle_observation_size=self._vehicle_observation_size,
            observation=observation,
            is_output_two_dimension=True,
        )
        self.generate_weights()
        # print("self.weights:", self._weights)
        return restart(observation=self._observation(), vehicle_observation=vehicle_observation, weights=self._weights)

    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """
        
        if self._reset_next_step:
            return self.reset()
        
        views_required_number, information_type_required_by_views_at_now, vehicle_actions, edge_action = \
            self.transform_action_array_to_actions(action)
        
        """Compute the baseline reward and difference rewards."""
        information_objects = self.compute_information_objects(
            views_required_number=views_required_number,
            information_type_required_by_views_at_now=information_type_required_by_views_at_now,
            vehicle_actions=vehicle_actions,
            edge_action=edge_action,
        )
        
        if self.is_reward_matrix:
            baseline_aov, baseline_cost, cumulative_aov, cumulative_cost, \
                average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, \
                average_sensing_cost, average_transmission_cost = self.compute_reward(
                    information_objects_ordered_by_views=information_objects[0],
                    vehicle_actions=vehicle_actions,
                )
            self._reward[-1][0] = baseline_aov
            self._reward[-1][1] = baseline_cost
            
            for i in range(self._config.vehicle_number):
                vehicle_baseline_aov, vehicle_baseline_cost, _, _, \
                    _, _, _, _, _, \
                    _, _ = self.compute_reward(
                    information_objects_ordered_by_views=information_objects[i + 1],
                    vehicle_actions=vehicle_actions,
                    vehicle_index=i,
                )
                self._reward[i][0] = -1 if baseline_aov - vehicle_baseline_aov < -1 else baseline_aov - vehicle_baseline_aov
                self._reward[i][0] = 1 if baseline_aov - vehicle_baseline_aov > 1 else baseline_aov - vehicle_baseline_aov
                
                self._reward[i][1] = -1 if baseline_cost - vehicle_baseline_cost < -1 else baseline_cost - vehicle_baseline_cost
                self._reward[i][1] = 1 if baseline_cost - vehicle_baseline_cost > 1 else baseline_cost - vehicle_baseline_cost
            
            aov_min, cost_min = self.get_min_reward_at_now(int(self._time_slots.now()))
            aov_max, cost_max = self.get_max_reward_at_now(int(self._time_slots.now()))
            if aov_min != 100 and aov_max != -100:
                if (aov_max - aov_min) == 0:
                    edge_aov = baseline_aov - aov_min
                else:
                    edge_aov = baseline_aov -  0.5 * (aov_max + aov_min)
            else:
                edge_aov = baseline_aov
            if cost_min != 100 and cost_max != -100:
                if (cost_max - cost_min) == 0:
                    edge_cost = baseline_cost - cost_min
                else:
                    edge_cost = baseline_cost - 0.5 * (cost_max + cost_min)
            else:
                edge_cost = baseline_cost
            
            self._reward[-2][0] = -1 if edge_aov < -1 else edge_aov
            self._reward[-2][0] = 1 if edge_aov > 1 else edge_aov
            self._reward[-2][1] = -1 if edge_cost < -1 else edge_cost
            self._reward[-2][1] = 1 if edge_cost > 1 else edge_cost

            """Update the information in the edge node."""
            self.update_information_in_edge(
                information_objects_ordered_by_views=information_objects[0],
            )
            # myapp.debug(f"\ninformation_objects_ordered_by_views:\n{self.string_of_information_objects_ordered_by_views(information_objects_ordered_by_views)}")
            observation = self._observation()

            vehicle_observation = self.get_vehicle_observations(
                vehicle_number=self._config.vehicle_number,
                information_number=self._config.information_number,
                sensed_information_number=self._config.sensed_information_number,
                vehicle_observation_size=self._vehicle_observation_size,
                observation=observation,
                is_output_two_dimension=True,
            )

            self.generate_weights()
            # print("self.weights:", self._weights)
            # check for termination
            if self._time_slots.is_end():
                self._reset_next_step = True
                return termination(observation=observation, reward=self._reward, vehicle_observation=vehicle_observation, weights=self._weights), cumulative_aov, cumulative_cost, \
                    average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, \
                    average_sensing_cost, average_transmission_cost
            self._time_slots.add_time()
            
            return transition(observation=observation, reward=self._reward, vehicle_observation=vehicle_observation, weights=self._weights), cumulative_aov, cumulative_cost, \
                average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, \
                average_sensing_cost, average_transmission_cost
    
        else:
            # print("reward: ", self.compute_reward(
            #         information_objects_ordered_by_views=information_objects[0],
            #         vehicle_actions=vehicle_actions,
            #     ))
            baseline_reward, cumulative_aov, cumulative_cost, average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, average_sensing_cost, average_transmission_cost = self.compute_reward(
                    information_objects_ordered_by_views=information_objects[0],
                    vehicle_actions=vehicle_actions,
                )
            self._reward[-1] = baseline_reward
            
            for i in range(self._config.vehicle_number):
                vehicle_baseline_reward, _, _, \
                    _, _, _, _, _, \
                    _, _ = self.compute_reward(
                    information_objects_ordered_by_views=information_objects[i + 1],
                    vehicle_actions=vehicle_actions,
                    vehicle_index=i,
                )
                vehicle_reward = baseline_reward - vehicle_baseline_reward
                self._reward[i] = vehicle_reward
            
            min_reward_history_at_now = self.get_min_reward_at_now(int(self._time_slots.now()))
            max_reward_history_at_now = self.get_max_reward_at_now(int(self._time_slots.now()))
            if min_reward_history_at_now != 100 and max_reward_history_at_now != -100:
                if (max_reward_history_at_now - min_reward_history_at_now) == 0:
                    edge_reward = baseline_reward - min_reward_history_at_now
                else:
                    edge_reward = (baseline_reward - min_reward_history_at_now) / (max_reward_history_at_now - min_reward_history_at_now)
            else:
                edge_reward = baseline_reward
            self._reward[-2] = edge_reward

            """Update the information in the edge node."""
            self.update_information_in_edge(
                information_objects_ordered_by_views=information_objects[0],
            )
            # myapp.debug(f"\ninformation_objects_ordered_by_views:\n{self.string_of_information_objects_ordered_by_views(information_objects_ordered_by_views)}")
            observation = self._observation()

            vehicle_observation = self.get_vehicle_observations(
                vehicle_number=self._config.vehicle_number,
                information_number=self._config.information_number,
                sensed_information_number=self._config.sensed_information_number,
                vehicle_observation_size=self._vehicle_observation_size,
                observation=observation,
                is_output_two_dimension=True,
            )

            # check for termination
            if self._time_slots.is_end():
                self._reset_next_step = True
                return termination(observation=observation, reward=self._reward, vehicle_observation=vehicle_observation), cumulative_aov, cumulative_cost, \
                    average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, \
                    average_sensing_cost, average_transmission_cost
            self._time_slots.add_time()
            return transition(observation=observation, reward=self._reward, vehicle_observation=vehicle_observation), cumulative_aov, cumulative_cost, \
                average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, \
                average_sensing_cost, average_transmission_cost

    def transform_action_array_to_actions(self, action: np.ndarray) -> Tuple[int, List[List[int]], List[vehicleAction], edgeAction]:
        """Transform the action array to the actions of vehicles and the edge node.
        Args:
            action: the action of the agent.
                which including the action of vehicles and the action of the edge.
                action[0: vehicle_number * vehicle_action_size] are the actions of vehicles.
                action[vehicle_number * vehicle_action_size: ] are the actions of the edge.
        Returns:
            actions: the actions of vehicles and the edge node.
        """ 
        
        vhielce_action_array = action[0: self._config.vehicle_number * self._vehicle_action_size]
        edge_action_array = action[self._config.vehicle_number * self._vehicle_action_size:]
        
        if len(vhielce_action_array) != self._config.vehicle_number * self._vehicle_action_size or \
            len(edge_action_array) != self._edge_action_size:
            raise ValueError('The length of the action is not correct.')

        vehicle_actions: List[vehicleAction] = [
            self.generate_vehicle_action_from_np_array(
                    vehicle_index=i,
                    now_time=self._time_slots.now(),
                    vehicle_list=self._vehicle_list,
                    information_list=self._information_list,
                    sensed_information_number=self._config.sensed_information_number,
                    network_output=vhielce_action_array[i * self._vehicle_action_size: (i + 1) * self._vehicle_action_size],
                    white_gaussian_noise=self._config.white_gaussian_noise,
                    edge_location=self._edge_node.get_edge_location(),
                    path_loss_exponent=self._config.path_loss_exponent,
                    SNR_target_low_bound=self._config.SNR_target_low_bound,
                    SNR_target_up_bound=self._config.SNR_target_up_bound,
                    probabiliity_threshold=self._config.probabiliity_threshold,
                    action_time=self._time_slots.now(),
            ) for i in range(self._config.vehicle_number)
        ]
        
        edge_action: edgeAction = self.generate_edge_action_from_np_array(
            now_time=self._time_slots.now(),
            edge_node=self._edge_node,
            action_time=self._time_slots.now(),
            network_output=edge_action_array,
            vehicle_number=self._config.vehicle_number,
        )
        
        views_required_number: int = self._information_requirements.get_views_required_number_at_now(self._time_slots.now())
        information_type_required_by_views_at_now: List[List[int]] = self._information_requirements.get_information_type_required_by_views_at_now_at_now(self._time_slots.now())
        
        return views_required_number, information_type_required_by_views_at_now, vehicle_actions, edge_action

    def compute_information_objects(
        self, 
        views_required_number: int,
        information_type_required_by_views_at_now: List[List[int]],
        vehicle_actions: List[vehicleAction],
        edge_action: edgeAction
    ) -> List[List[List[informationPacket]]]:
        """Compute the reward.
        Args:
            views_required_number: the number of views required by applications.
            information_type_required_by_views_at_now: the information type required by the views.
            vehicle_actions: the actions of the vehicle.
            edge_action: the action of the edge node.
            vehicle_index: the index of the vehicle which do nothing, i.e., its action is null.
                the default value is -1, which means no vehicles do nothing.
        Returns:
            the information objects: the objects under all vehicles and one vehicle do nothing.
            (i.e., returns[0] is the information objects under all vehicles and returns[1] is the information objects under vehicle v1 do nothing.)
        """

        information_objects_ordered_by_views: List[List[List[informationPacket]]] = []
        for vehicle_index in range(self._config.vehicle_number + 1):
            information_objects_ordered_by_views.append(list())
            for _ in range(views_required_number):
                information_objects_ordered_by_views[vehicle_index].append(list())

        for i in range(self._config.vehicle_number):
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
                mean_channel_fading_gain=self._config.mean_channel_fading_gain,
                second_moment_channel_fading_gain=self._config.second_moment_channel_fading_gain,
                path_loss_exponent=self._config.path_loss_exponent,
                information_list=self._information_list,
            )

            transmission_times = v2i_transmission.get_transmission_times()

            for information_index in range(len(sensed_information_type)):
                infor = informationPacket(
                    type=sensed_information_type[information_index],
                    vehicle_index=i,
                    edge_index=self._edge_node.get_edge_index(),
                    updating_moment=updating_moments[information_index],
                    inter_arrival_interval=arrival_intervals[information_index],
                    arrival_moment=arrival_moments[information_index],
                    queuing_time=queuing_times[information_index],
                    transmission_time=transmission_times[information_index],
                    received_moment=arrival_moments[information_index] + queuing_times[information_index] + transmission_times[information_index],
                )

                for view_index in range(len(information_type_required_by_views_at_now)):
                    if infor.get_type() in information_type_required_by_views_at_now[view_index]:
                        information_objects_ordered_by_views[0][view_index].append(infor)
                        for vehicle_index in range(self._config.vehicle_number):
                            if i == vehicle_index:
                                pass
                            else:
                                information_objects_ordered_by_views[vehicle_index + 1][view_index].append(infor)
                        
        """If the view is uncomplete, add the missing information from the information in edge to the view."""
        for vehicle_index in range(self._config.vehicle_number + 1):
            for view_index in range(len(information_type_required_by_views_at_now)):
                for infor_type in information_type_required_by_views_at_now[view_index]:
                    infor_type_exist = False
                    for infor in information_objects_ordered_by_views[vehicle_index][view_index]:
                        if infor.get_type() == infor_type:
                            infor_type_exist = True
                            break
                    if not infor_type_exist and len(self._information_in_edge[infor_type]) > 0:
                        infor_in_edge = self._information_in_edge[infor_type][0]
                        information_objects_ordered_by_views[vehicle_index][view_index].append(infor_in_edge)

        return information_objects_ordered_by_views

    def update_information_in_edge(self, information_objects_ordered_by_views: List[List[informationPacket]]) -> None:
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
        information_objects_ordered_by_views: List[List[informationPacket]],
        vehicle_actions: List[vehicleAction],
        vehicle_index: int = -1
    ):
        """
            return:
                the reward of the different dim.
        """
        cumulative_aov = 0
        cumulative_cost = 0
        average_aov = 0
        average_cost = 0
        average_timeliness = 0
        average_consistency = 0
        average_redundancy = 0
        average_sensing_cost = 0
        average_transmission_cost = 0

        """Compute the timeliness of views"""
        timeliness_views = []
        for information_objects in information_objects_ordered_by_views:
            timeliness_list = []
            for _ in range(self._config.vehicle_number):
                timeliness_list.append(list())
            for infor in information_objects:
                # if infor.get_arrival_moment() + infor.get_queuing_time() + infor.get_transmission_time() - infor.get_updating_moment() > 1000:
                #     print("infor.get_arrival_moment(): ", infor.get_arrival_moment())
                #     print("infor.get_queuing_time(): ", infor.get_queuing_time())
                #     print("infor.get_transmission_time(): ", infor.get_transmission_time())
                #     print("infor.get_updating_moment(): ", infor.get_updating_moment())
                #     print("*****************************************************************************************")
                timeliness_list[infor.get_vehicle_index()].append(
                    infor.get_arrival_moment() + infor.get_queuing_time() + infor.get_transmission_time() - infor.get_updating_moment()
                
                )
            timeliness_of_vehicles = []
            for values in timeliness_list:
                if values == []:
                    timeliness_of_vehicles.append(0)
                else:
                    timeliness_of_vehicles.append(max(values))
            timeliness = max(timeliness_of_vehicles)
            if not np.isinf(timeliness) and not np.isnan(timeliness):
                timeliness_views.append(timeliness)
                if timeliness > self._max_timeliness:
                    self._max_timeliness = timeliness
                if timeliness < self._min_timeliness:
                    self._min_timeliness = timeliness
        
        """Compute the consistency of views"""
        consistency_views = []
        for information_objects in information_objects_ordered_by_views:
            updating_moments_of_informations = []
            for infor in information_objects:
                updating_moments_of_informations.append(infor.get_updating_moment())
            consistency = max(updating_moments_of_informations) - min(updating_moments_of_informations)
            # if consistency > 100:
            #     print("max(updating_moments_of_informations): ", max(updating_moments_of_informations))
            #     print("min(updating_moments_of_informations): ", min(updating_moments_of_informations))
            #     print("*****************************************************************************************")
            if not np.isinf(consistency) and not np.isnan(consistency):
                consistency_views.append(consistency)
                if consistency > self._max_consistency:
                    self._max_consistency = consistency
                if consistency < self._min_consistency:
                    self._min_consistency = consistency


        """Compute the redundancy of views"""
        redundancy_views = []
        for information_objects in information_objects_ordered_by_views:
            redundancy = 0
            redundancy_list = []
            for _ in range(self._config.information_number):
                redundancy_list.append(list())
            for infor in information_objects:
                redundancy_list[int(infor.get_type())].append(1)
            for i in range(self._config.information_number):
                if sum(redundancy_list[i]) > 1:
                    redundancy += sum(redundancy_list[i]) - 1
            if not np.isinf(redundancy) and not np.isnan(redundancy):
                redundancy_views.append(redundancy)
                if redundancy > self._max_redundancy:
                    self._max_redundancy = redundancy
                if redundancy < self._min_redundancy:
                    self._min_redundancy = redundancy
        
        """Compute the cost of view"""
        sensing_cost_views = []
        transmission_cost_views = []
        for information_objects in information_objects_ordered_by_views:
            sensing_cost_list = []
            transmission_cost_list = []
            for _ in range(self._config.vehicle_number):
                sensing_cost_list.append(list())
                transmission_cost_list.append(list())
            for infor in information_objects:
                if infor.get_vehicle_index() != -1:
                    sensing_cost_list[infor.get_vehicle_index()].append(
                        self._vehicle_list.get_vehicle(infor.get_vehicle_index()).get_sensing_cost_by_type(infor.get_type()) 
                    )
                    # if infor.get_transmission_time() * vehicle_actions[infor.get_vehicle_index()].get_transmission_power() > 100:
                    #     print("infor.get_transmission_time() * vehicle_actions[infor.get_vehicle_index()].get_transmission_power(): ", infor.get_transmission_time() * vehicle_actions[infor.get_vehicle_index()].get_transmission_power())
                    #     print("infor.get_transmission_time(): ", infor.get_transmission_time())
                    #     print("vehicle_actions[infor.get_vehicle_index()].get_transmission_power(): ", vehicle_actions[infor.get_vehicle_index()].get_transmission_power())
                    #     print("*****************************************************************************************")
                    transmission_cost_list[infor.get_vehicle_index()].append(
                        infor.get_transmission_time() * vehicle_actions[infor.get_vehicle_index()].get_transmission_power()
                    )
            sensing_cost_of_vehicles = []
            transmission_cost_of_vehicles = []
            for values in sensing_cost_list:
                if values == []:
                    sensing_cost_of_vehicles.append(0)
                else:
                    sensing_cost_of_vehicles.append(sum(values) / len(values))
            for values in transmission_cost_list:
                if values == []:
                    transmission_cost_of_vehicles.append(0)
                else:
                    # print("sum(values) / len(values): ", sum(values) / len(values))
                    if sum(values) / len(values) > 50:
                        transmission_cost_of_vehicles.append(50)
                    else:
                        transmission_cost_of_vehicles.append(sum(values) / len(values))
            
            sensing_cost = sum(sensing_cost_of_vehicles) / len(sensing_cost_of_vehicles)
            if not np.isinf(sensing_cost) and not np.isnan(sensing_cost):
                sensing_cost_views.append(sensing_cost)
                if sensing_cost > self._max_sensing_cost:
                    self._max_sensing_cost = sensing_cost
                if sensing_cost < self._min_sensing_cost:
                    self._min_sensing_cost = sensing_cost
            transmission_cost = sum(transmission_cost_of_vehicles) / len(transmission_cost_of_vehicles)
            if not np.isinf(transmission_cost) and not np.isnan(transmission_cost):
                transmission_cost_views.append(transmission_cost)
                if transmission_cost > self._max_transmission_cost:
                    self._max_transmission_cost = transmission_cost
                if transmission_cost < self._min_transmission_cost:
                    self._min_transmission_cost = transmission_cost
        
        """Normalize the timeliness, consistency, redundancy, and cost of views"""
        timeliness_views_normalized = []
        consistency_views_normalized = []
        redundancy_views_normalized = []
        sensing_cost_views_normalized = []
        transmission_cost_views_normalized = []
        
        for i in range(len(timeliness_views)):
            if self._max_timeliness != -1 and self._min_timeliness != 100000000 and (self._max_timeliness - self._min_timeliness) != 0 and  \
                not np.isnan(timeliness_views[i] - self._min_timeliness) / (self._max_timeliness - self._min_timeliness):

                timeliness_views_normalized.append(
                    (timeliness_views[i] - self._min_timeliness) / (self._max_timeliness - self._min_timeliness)
                )
            else:
                timeliness_views_normalized.append(-1)
            
            if self._max_consistency != -1 and self._min_consistency != 100000000 and (self._max_consistency - self._min_consistency) != 0 and \
                not np.isnan(consistency_views[i] - self._min_consistency) / (self._max_consistency - self._min_consistency):

                consistency_views_normalized.append(
                    (consistency_views[i] - self._min_consistency) / (self._max_consistency - self._min_consistency)
                )
            else:
                consistency_views_normalized.append(-1)
            
            if self._max_redundancy != -1 and self._min_redundancy != 100000000 and (self._max_redundancy - self._min_redundancy) != 0 and \
                not np.isnan((redundancy_views[i] - self._min_redundancy) / (self._max_redundancy - self._min_redundancy)):

                redundancy_views_normalized.append(
                    (redundancy_views[i] - self._min_redundancy) / (self._max_redundancy - self._min_redundancy)
                )
            else:
                redundancy_views_normalized.append(-1)

            if self._max_sensing_cost != -1 and self._min_sensing_cost != 100000000 and (self._max_sensing_cost - self._min_sensing_cost) != 0 and \
                not np.isnan(sensing_cost_views[i] - self._min_sensing_cost) / (self._max_sensing_cost - self._min_sensing_cost):
                sensing_cost_views_normalized.append(
                    (sensing_cost_views[i] - self._min_sensing_cost) / (self._max_sensing_cost - self._min_sensing_cost)
                )
            else:
                sensing_cost_views_normalized.append(-1)
                
            if self._max_transmission_cost != -1 and self._min_transmission_cost != 100000000 and (self._max_transmission_cost - self._min_transmission_cost) != 0 and \
                not np.isnan(transmission_cost_views[i] - self._min_transmission_cost) / (self._max_transmission_cost - self._min_transmission_cost):
                transmission_cost_views_normalized.append(
                    (transmission_cost_views[i] - self._min_transmission_cost) / (self._max_transmission_cost - self._min_transmission_cost)
                )
            else:
                transmission_cost_views_normalized.append(-1)
        # print("************************************************************************************************************************")
        # print("max_timeliness: ", self._max_timeliness)
        # print("max_consistency: ", self._max_consistency)
        # print("max_redundancy: ", self._max_redundancy)
        # print("max_sensing_cost: ", self._max_sensing_cost)
        # print("max_transmission_cost: ", self._max_transmission_cost)
        # print("timeliness_views_normalized: ", timeliness_views_normalized)
        # print("consistency_views_normalized: ", consistency_views_normalized)
        # print("redundancy_views_normalized: ", redundancy_views_normalized)
        # print("sensing_cost_views_normalized: ", sensing_cost_views_normalized)
        # print("transmission_cost_views_normalized: ", transmission_cost_views_normalized)

        if len(timeliness_views_normalized) > 0:        
            average_timeliness = sum(timeliness_views_normalized) / len(timeliness_views_normalized)
        if len(consistency_views_normalized) > 0:
            average_consistency = sum(consistency_views_normalized) / len(consistency_views_normalized)
        if len(redundancy_views_normalized) > 0:
            average_redundancy = sum(redundancy_views_normalized) / len(redundancy_views_normalized)
        if len(sensing_cost_views_normalized) > 0:
            average_sensing_cost = sum(sensing_cost_views_normalized) / len(sensing_cost_views_normalized)
        if len(transmission_cost_views_normalized) > 0:
            average_transmission_cost = sum(transmission_cost_views_normalized) / len(transmission_cost_views_normalized)
    
        if self.is_reward_matrix:
        
            """Compute the age of view."""
            age_of_view = []
            cost_of_view = []
            for i in range(len(timeliness_views_normalized)):
                if timeliness_views_normalized[i] != -1 and consistency_views_normalized[i] != -1:
                    age_of_view.append(
                        self._config.weight_of_timeliness * timeliness_views_normalized[i] + \
                        self._config.weight_of_consistency * consistency_views_normalized[i]
                    )
                if  redundancy_views_normalized[i] != -1 and sensing_cost_views_normalized[i] != -1 and transmission_cost_views_normalized[i] != -1:
                    cost_of_view.append(
                        self._config.weight_of_redundancy * redundancy_views_normalized[i] + \
                        self._config.weight_of_sensing_cost * sensing_cost_views_normalized[i] + \
                        self._config.weight_of_tranmission_cost * transmission_cost_views_normalized[i]
                    )
                    
            if len(age_of_view) > 0:
                average_aov = sum(age_of_view) / len(age_of_view)
            if len(cost_of_view) > 0:
                average_cost = sum(cost_of_view) / len(cost_of_view)
            
            """Normalize the age of view."""
            if len(age_of_view) > 0:
                normalized_age_of_view = float(1.0 - sum(age_of_view) / len(age_of_view))
            else:
                normalized_age_of_view = 0
            normalized_age_of_view = 0 if normalized_age_of_view < 0 else normalized_age_of_view
            normalized_age_of_view = 1 if normalized_age_of_view > 1 else normalized_age_of_view
            
            """Normalize the cost of view."""
            if len(cost_of_view) > 0:
                normalized_cost_of_view = float(1.0 - sum(cost_of_view) / len(cost_of_view))
            else:
                normalized_cost_of_view = 0
            normalized_cost_of_view = 0 if normalized_cost_of_view < 0 else normalized_cost_of_view
            normalized_cost_of_view = 1 if normalized_cost_of_view > 1 else normalized_cost_of_view
            
            cumulative_aov = normalized_age_of_view
            cumulative_cost = normalized_cost_of_view
            
            if vehicle_index == -1:
                self.append_reward_at_now(
                    now=int(self._time_slots.now()),
                    aov=normalized_age_of_view,
                    cost=normalized_cost_of_view,
                )
            
            return normalized_age_of_view, normalized_cost_of_view, cumulative_aov, cumulative_cost, \
                average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, \
                average_sensing_cost, average_transmission_cost

        else:
            """Compute the age of view."""
            age_of_view = []
            cost_of_view = []
            for i in range(len(timeliness_views_normalized)):
                if timeliness_views_normalized[i] != -1 and consistency_views_normalized[i] != -1:
                    age_of_view.append(
                        self._config.weight_of_timeliness * timeliness_views_normalized[i] + \
                        self._config.weight_of_consistency * consistency_views_normalized[i]
                    )
                if  redundancy_views_normalized[i] != -1 and sensing_cost_views_normalized[i] != -1 and transmission_cost_views_normalized[i] != -1:
                    cost_of_view.append(
                        self._config.weight_of_redundancy * redundancy_views_normalized[i] + \
                        self._config.weight_of_sensing_cost * sensing_cost_views_normalized[i] + \
                        self._config.weight_of_tranmission_cost * transmission_cost_views_normalized[i]
                    )
                    
            if len(age_of_view) > 0:
                average_aov = sum(age_of_view) / len(age_of_view)
            if len(cost_of_view) > 0:
                average_cost = sum(cost_of_view) / len(cost_of_view)
            
            """Normalize the age of view."""
            if len(age_of_view) > 0:
                normalized_age_of_view = float(1.0 - sum(age_of_view) / len(age_of_view))
            else:
                normalized_age_of_view = 0
            normalized_age_of_view = 0 if normalized_age_of_view < 0 else normalized_age_of_view
            normalized_age_of_view = 1 if normalized_age_of_view > 1 else normalized_age_of_view
            
            """Normalize the cost of view."""
            if len(cost_of_view) > 0:
                normalized_cost_of_view = float(1.0 - sum(cost_of_view) / len(cost_of_view))
            else:
                normalized_cost_of_view = 0
            normalized_cost_of_view = 0 if normalized_cost_of_view < 0 else normalized_cost_of_view
            normalized_cost_of_view = 1 if normalized_cost_of_view > 1 else normalized_cost_of_view
            
            cumulative_aov = normalized_age_of_view
            cumulative_cost = normalized_cost_of_view
            
            
            """Compute the age of view."""
            age_of_view = []
            for i in range(len(timeliness_views_normalized)):
                if timeliness_views_normalized[i] != -1 and consistency_views_normalized[i] != -1 and redundancy_views_normalized[i] != -1 and sensing_cost_views_normalized[i] != -1 and transmission_cost_views_normalized[i] != -1:
                    age_of_view.append(
                        self._config.static_weight_of_timeliness * timeliness_views_normalized[i] + \
                        self._config.static_weight_of_consistency * consistency_views_normalized[i] + \
                        self._config.static_weight_of_redundancy * redundancy_views_normalized[i] + \
                        self._config.static_weight_of_sensing_cost * sensing_cost_views_normalized[i] + \
                        self._config.static_weight_of_tranmission_cost * transmission_cost_views_normalized[i]
                    )

            if len(age_of_view) > 0:
                reward = float(1.0 - sum(age_of_view) / len(age_of_view))
            else:
                reward = 0
            reward = 0 if reward < 0 else reward
            reward = 1 if reward > 1 else reward

            if vehicle_index == -1:
                self.append_reward_at_now(
                    now=int(self._time_slots.now()),
                    reward=reward,
                )
            return reward, cumulative_aov, cumulative_cost, \
                average_aov, average_cost, average_timeliness, average_consistency, average_redundancy, \
                average_sensing_cost, average_transmission_cost

    """Define the observation spaces of vehicle."""
    def vehicle_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(self._vehicle_observation_size,),
            dtype=float,
            minimum=np.zeros((self._vehicle_observation_size,)),
            maximum=np.ones((self._vehicle_observation_size,)),
            name='vehicle_observations'
        )
    
    """Define the observation spaces of vehicles."""
    def vehicle_all_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(self._config.vehicle_number, self._vehicle_observation_size),
            dtype=float,
            minimum=np.zeros((self._config.vehicle_number , self._vehicle_observation_size)),
            maximum=np.ones((self._config.vehicle_number , self._vehicle_observation_size)),
            name='vehicle_all_observations'
        )

    """Define the action spaces of vehicle."""
    def vehicle_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._vehicle_action_size,),
            dtype=float,
            minimum=np.zeros((self._vehicle_action_size,)),
            maximum=np.ones((self._vehicle_action_size,)),
            name='vehicle_actions'
        )

    """Define the action spaces of vehicle in critic network."""
    def vehicle_critic_network_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._vehicle_critic_network_action_size,),
            dtype=float,
            minimum=np.zeros((self._vehicle_critic_network_action_size,)),
            maximum=np.ones((self._vehicle_critic_network_action_size,)),
            name='critic_vehicle_actions'
        )
    
    def vehicle_critic_network_other_action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self._vehicle_critic_network_action_size - self._vehicle_action_size,),
            dtype=float,
            minimum=np.zeros((self._vehicle_critic_network_action_size - self._vehicle_action_size,)),
            maximum=np.ones((self._vehicle_critic_network_action_size - self._vehicle_action_size,)),
            name='critic_vehicle_other_actions'
        )
    
    
    """Define the observation spaces of edge."""
    def edge_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(self._edge_observation_size,),
            dtype=float,
            minimum=np.zeros((self._edge_observation_size,)),
            maximum=np.ones((self._edge_observation_size,)),
            name='edge_observations'
        )

    """Define the action spaces of edge."""
    def edge_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._edge_action_size,),
            dtype=float,
            minimum=np.zeros((self._edge_action_size,)),
            maximum=np.ones((self._edge_action_size,)),
            name='edge_actions'
        )

    """Define the action spaces of edge in critic network."""
    def edge_critic_network_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._edge_critic_network_action_size,),
            dtype=float,
            minimum=np.zeros((self._edge_critic_network_action_size,)),
            maximum=np.ones((self._edge_critic_network_action_size,)),
            name='critic_edge_actions',
        )
    

    """Define the gloabl observation spaces."""
    def observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        return specs.BoundedArray(
            shape=(self._observation_size,),
            dtype=float,
            minimum=np.zeros((self._observation_size,)),
            maximum=np.ones((self._observation_size,)),
            name='observations'
        )
    
    """Define the gloabl action spaces."""
    def action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        return specs.BoundedArray(
            shape=(self._action_size,),
            dtype=float,
            minimum=np.zeros((self._action_size,)),
            maximum=np.ones((self._action_size,)),
            name='actions'
        )

    def reward_spec(self):
        """Define and return the reward space."""
        if self.is_reward_matrix:
            return specs.Array(
                shape=(self._reward_size, self._config.weighting_number), 
                dtype=float, 
                name='rewards'
            )
        else:
            return specs.Array(
                shape=(self._reward_size,), 
                dtype=float, 
                name='rewards'
            )
    
    def weights_spec(self):
        """Define and return the weight space."""
        return specs.BoundedArray(
            shape=(self._config.weighting_number, ), 
            dtype=float, 
            minimum=np.zeros((self._config.weighting_number, )),
            maximum=np.ones((self._config.weighting_number, )),
            name='weights'
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
        # 1
        observation[index] = float(self._time_slots.now() / self._time_slots.get_number())
        index += 1
        # vehicle distances
        # 2 - 11
        for _ in range(self._config.vehicle_number):
            observation[index] = float(self._vehicle_list.get_vehicle(_).get_distance_between_edge(
                nowTimeSlot=self._time_slots.now(),
                edge_location=self._edge_node.get_edge_location(),
            ) / (self._edge_node.get_communication_range() * np.sqrt(2)))
            if observation[index] > 1:
                observation[index] = 1
            index += 1
        # information_canbe_senseds
        # 12 - 41
        for _ in range(self._config.vehicle_number):
            for __ in range(self._config.sensed_information_number):
                observation[index] = float(self._vehicle_list.get_vehicle(_).get_information_canbe_sensed()[__]
                    / self._config.information_number)
                index += 1
        # sensing_cost_of_informations
        # 42 - 71
        for _ in range(self._config.vehicle_number):
            for __ in range(self._config.sensed_information_number):
                observation[index] = float(self._vehicle_list.get_vehicle(_).get_sensing_cost()[__]
                    / self._config.max_sensing_cost)
                index += 1
        # information_in_edge
        # 72 - 81
        for _ in range(self._config.information_number):
            if len(self._information_in_edge[_]) == 0:
                observation[index] = 0
            else:
                if self._information_in_edge[_][0].get_received_moment() >= self._time_slots.get_number():
                    observation[index] = 1
                else:
                    observation[index] = float(self._information_in_edge[_][0].get_received_moment() 
                        / self._time_slots.get_number())
            index += 1
        # information_requried
        # 82 - 91
        for _ in range(self._config.information_number):
            observation[index] = float(self._information_requirements.get_information_required_at_now(self._time_slots.now())[_])
            index += 1
        return observation

    @staticmethod
    def get_vehicle_observations(
        vehicle_number: int, 
        information_number: int, 
        sensed_information_number: int, 
        vehicle_observation_size: int, 
        observation: np.ndarray,
        is_output_two_dimension: bool = True,    
    ) -> np.ndarray:
        """Return the observations of the environment at each vehicle.
        vehicle_observation_size: int = 1 + 1 + 1 + self._config.sensed_information_number + self._config.sensed_information_number + \
            self._config.information_number + self._config.information_number 
            # now_time_slot + vehicle_index + distance + information_canbe_sensed + sensing_cost_of_information + \
            # information_in_edge + information_requried
        """
        vehicle_observations = np.zeros(shape=(vehicle_number, vehicle_observation_size))
        
        index = 0
        observation_index = 0
        # now_time_slot
        for _ in range(vehicle_number):
            vehicle_observations[_ , index] = observation[observation_index]
        index += 1
        observation_index += 1

        # vehicle_index
        for _ in range(vehicle_number):
            vehicle_observations[_ , index] = float(_ / vehicle_number)
        index += 1

        # vehicle distances
        for _ in range(vehicle_number):
            vehicle_observations[_ , index] = observation[observation_index]
            observation_index += 1
        index += 1

        # information_canbe_senseds
        for _ in range(vehicle_number):
            origin_index = index
            for __ in range(sensed_information_number):
                vehicle_observations[_ , origin_index] = observation[observation_index]
                observation_index += 1
                origin_index += 1
        index = origin_index

        # sensing_cost_of_informations
        for _ in range(vehicle_number):
            origin_index = index
            for __ in range(sensed_information_number):
                vehicle_observations[_ , origin_index] = observation[observation_index]
                observation_index += 1
                origin_index += 1
        index = origin_index

        # information_in_edge
        for _ in range(information_number):
            for __ in range(vehicle_number):
                vehicle_observations[__ , index] = observation[observation_index]
            observation_index += 1
            index += 1
        
        # information_requried
        for _ in range(information_number):
            for __ in range(vehicle_number):
                vehicle_observations[__ , index] = observation[observation_index]
            observation_index += 1
            index += 1

        """flatten the output to fit the learning, i.e., transitions.vehicle_observation in sample.data
        should be M * [vehicle_number * vehicle_observation_size]
        Code is:
            new_vehicle_observations = np.zeros(shape=(vehicle_number*vehicle_observation_size,))
            index = 0
            for _ in range(vehicle_number):
                for __ in range(vehicle_observation_size):
                    new_vehicle_observations[index] = vehicle_observations[_ , __]
                    index += 1
            return new_vehicle_observations
        """
        if not is_output_two_dimension:
            return vehicle_observations.flatten()

        return vehicle_observations

    @staticmethod
    def get_edge_observation(observation: np.ndarray) -> np.ndarray:
        """Return the observation of the environment at edge."""
        return observation

    def string_of_information_objects_ordered_by_views(self, information_objects_ordered_by_views: List[List[informationPacket]]) -> str:
        """Return the string of information objects ordered by views."""
        string = ""
        for _ in range(len(information_objects_ordered_by_views)):
            string += "view" + str(_) + ": "
            for __ in range(len(information_objects_ordered_by_views[_])):
                string += str(information_objects_ordered_by_views[_][__]) + " "
            string += "\n"
        return string

    def compute_SNR(
        self, 
        white_gaussian_noise: int,
        channel_fading_gain: float,
        distance: float,
        path_loss_exponent: int,
        transmission_power: float) -> float:
        """
        Compute the SNR of a vehicle transmission
        Args:
            white_gaussian_noise: the white gaussian noise of the channel, e.g., -70 dBm
            channel_fading_gain: the channel fading gain, e.g., Gaussion distribution with mean 2 and variance 0.4
            distance: the distance between the vehicle and the edge, e.g., 300 meters
            path_loss_exponent: the path loss exponent, e.g., 3
            transmission_power: the transmission power of the vehicle, e.g., 10 mW
        Returns:
            SNR: the SNR of the transmission
        """
        
        return (1.0 / (np.power(10, (white_gaussian_noise / 10)) / 1000)) * \
            np.power(np.abs(channel_fading_gain), 2) * \
            1.0 / (np.power(distance, path_loss_exponent)) * \
            transmission_power / 1000

    def compute_successful_tansmission_probability(
        self, 
        white_gaussian_noise: int,
        distance: float,
        path_loss_exponent: int,
        transmission_power: float,
        SNR_target: float) -> float:
        """
        Compute the sussessful transmission probability of the vehicle to the edge
        Args:
            white_gaussian_noise: the white gaussian noise of the channel
            channel_fading_gains: the channel fading gains
            distance: the distance between the vehicle and the edge
            path_loss_exponent: the path loss exponent
            transmission_power: the transmission power of the vehicle
            SNR_target: the target SNR
        Returns:
            sussessful_tansmission_probability: the sussessful transmission probability of the vehicle to the edge
        """
        hash_id = str(hash(str(distance) + str(transmission_power)))
        if self._successful_tansmission_probability.get(hash_id) is None:
            successful_transmission_number = 0
            total_number = 0
            for channel_fading_gain in self._channel_fading_gains:
                total_number += 1
                SNR = self.compute_SNR(
                    white_gaussian_noise=white_gaussian_noise,
                    channel_fading_gain=channel_fading_gain,
                    distance=distance,
                    path_loss_exponent=path_loss_exponent,
                    transmission_power=transmission_power
                )
                # print("distance: " + str(distance) + " transmission_power: " + str(transmission_power) + " channel_fading_gain: " + str(channel_fading_gain) + " SNR: " + str(SNR))
                # print("SNR: ", SNR)
                # print("SNR_target: ", SNR_target)
                # print("SNR value: ", 10 * np.log10(SNR))
                if SNR != 0 and 10 * np.log10(SNR) >= SNR_target:
                    successful_transmission_number += 1
            self._successful_tansmission_probability[hash_id] = successful_transmission_number / total_number
            return successful_transmission_number / total_number
        else:
            return self._successful_tansmission_probability[hash_id]

    def generate_channel_fading_gain(self, mean_channel_fading_gain, second_moment_channel_fading_gain, size: int = 1):
        channel_fading_gain = np.random.normal(loc=mean_channel_fading_gain, scale=second_moment_channel_fading_gain, size=size)
        return channel_fading_gain

    def get_minimum_transmission_power(
        self, 
        white_gaussian_noise: int,
        distance: float,
        path_loss_exponent: int,
        transmission_power: float,
        SNR_target: float,
        probabiliity_threshold: float) -> float:
        """
        Get the minimum transmission power of the vehicle to the edge
        Args:
            white_gaussian_noise: the white gaussian noise of the channel
            mean_channel_fading_gain: the mean channel fading gain
            second_moment_channel_fading_gain: the second moment channel fading gain
            distance: the distance between the vehicle and the edge
            path_loss_exponent: the path loss exponent
            transmission_power: the transmission power of the vehicle
            SNR_target: the target SNR
            probabiliity_threshold: the probability threshold
        Returns:
            minimum_transmission_power: the minimum transmission power of the vehicle to the edge
        """

        minimum_transmission_power = transmission_power
        minimum_power = 1
        maximum_power = transmission_power
        while_flag = True
        mid = minimum_power
        mid_probabiliity = self.compute_successful_tansmission_probability(
            white_gaussian_noise=white_gaussian_noise,
            distance=distance,
            path_loss_exponent=path_loss_exponent,
            transmission_power=mid,
            SNR_target=SNR_target
        )
        if mid_probabiliity > probabiliity_threshold:
            minimum_transmission_power = mid
            while_flag = False
        
        mid = maximum_power
        mid_probabiliity = self.compute_successful_tansmission_probability(
            white_gaussian_noise=white_gaussian_noise,
            distance=distance,
            path_loss_exponent=path_loss_exponent,
            transmission_power=mid,
            SNR_target=SNR_target
        )
        if mid_probabiliity <= probabiliity_threshold:
            # print("mid_probabiliity: " + str(mid_probabiliity))
            # print("probabiliity_threshold: " + str(probabiliity_threshold))
            
            minimum_transmission_power = mid
            while_flag = False
        
        while while_flag:
            mid = (minimum_power + maximum_power) / 2
            mid_plus = mid + 1
            mid_probabiliity = self.compute_successful_tansmission_probability(
                white_gaussian_noise=white_gaussian_noise,
                distance=distance,
                path_loss_exponent=path_loss_exponent,
                transmission_power=mid,
                SNR_target=SNR_target
            )
            mid_plus_probabiliity = self.compute_successful_tansmission_probability(
                white_gaussian_noise=white_gaussian_noise,
                distance=distance,
                path_loss_exponent=path_loss_exponent,
                transmission_power=mid_plus,
                SNR_target=SNR_target
            )

            if minimum_power > maximum_power:
                minimum_transmission_power = (minimum_power + maximum_power) / 2
                break
            if mid_probabiliity <= probabiliity_threshold and mid_plus_probabiliity >= probabiliity_threshold:
                minimum_transmission_power = mid
                break
            else:
                if mid_probabiliity < probabiliity_threshold:
                    minimum_power = mid
                else:
                    maximum_power = mid

        return minimum_transmission_power


    def rescale_the_list_to_small_than_one(self, list_to_rescale: List[float], is_sum_equal_one: Optional[bool] = False) -> List[float]:
        """ rescale the list small than one.
        Args:
            list_to_rescale: list to rescale.
        Returns:
            rescaled list.
        """
        if is_sum_equal_one:
            maximum_sum = sum(list_to_rescale)
        else:
            maximum_sum = sum(list_to_rescale) + 0.00001
        return [x / maximum_sum for x in list_to_rescale]   # rescale the list to small than one.

    def generate_vehicle_action_from_np_array(
        self, 
        now_time: int,
        vehicle_index: int,
        vehicle_list: vehicleList,
        information_list: informationList,
        sensed_information_number: int,
        network_output: np.ndarray,
        white_gaussian_noise: int,
        edge_location: location,
        path_loss_exponent: int,
        SNR_target_low_bound: float,
        SNR_target_up_bound: float,
        probabiliity_threshold: float,
        action_time: int):
        """ generate the vehicle action from the neural network output.

        self._vehicle_action_size = self._sensed_information_number + self._sensed_information_number + \
            self._sensed_information_number + 1
            # sensed_information + sensing_frequencies + uploading_priorities + transmission_power

        Args:
            network_output: the output of the neural network.
        Returns:
            the vehicle action.
        """
        sensed_information = np.zeros(sensed_information_number)
        sensing_frequencies = np.zeros(sensed_information_number)
        uploading_priorities = np.zeros(sensed_information_number)

        for index, values in enumerate(network_output[:sensed_information_number]):
            if values > 0.5:
                sensed_information[index] = 1
        frequencies = network_output[sensed_information_number: 2*sensed_information_number]
        frequencies = self.rescale_the_list_to_small_than_one(frequencies)
        for index, values in enumerate(frequencies):
            if sensed_information[index] == 1:
                sensing_frequencies[index] = values / information_list.get_mean_service_time_by_vehicle_and_type(
                    vehicle_index=vehicle_index,
                    data_type_index=vehicle_list.get_vehicle(vehicle_index).get_information_type_canbe_sensed(index)
                )
                # if sensing_frequencies[index] < 0.01:
                #     print("sensing_frequencies[index]: ", sensing_frequencies[index])
                #     print("values: ", values)
        for index, values in enumerate(network_output[2*sensed_information_number: 3*sensed_information_number]):
            if sensed_information[index] == 1:
                uploading_priorities[index] = values

        sensed_information = list(sensed_information)
        sensing_frequencies = list(sensing_frequencies)
        uploading_priorities = list(uploading_priorities)

        SNR_target = np.random.random() * (SNR_target_up_bound - SNR_target_low_bound) + SNR_target_low_bound

        minimum_transmission_power = self.get_minimum_transmission_power(
            white_gaussian_noise=white_gaussian_noise,
            distance=vehicle_list.get_vehicle(vehicle_index).get_vehicle_location(now_time).get_distance(edge_location),
            path_loss_exponent=path_loss_exponent,
            transmission_power=vehicle_list.get_vehicle(vehicle_index).get_transmission_power(),
            SNR_target=SNR_target,
            probabiliity_threshold=probabiliity_threshold
        )
        # print("minimum_transmission_power: ", minimum_transmission_power)

        transmisson_power = minimum_transmission_power + network_output[-1] * \
            (vehicle_list.get_vehicle(vehicle_index).get_transmission_power() - minimum_transmission_power)
        
        vehicle_action = vehicleAction(
            vehicle_index=vehicle_index,
            now_time=now_time,

            sensed_information=sensed_information,
            sensing_frequencies=sensing_frequencies,
            uploading_priorities=uploading_priorities,
            transmission_power=transmisson_power,

            action_time=action_time,
        )

        if not vehicle_action.check_action(now_time, vehicle_list):
            raise ValueError("The vehicle action is not valid.")
        
        return vehicle_action


    def generate_edge_action_from_np_array(
        self, 
        now_time: int,
        edge_node: edge,
        action_time: int,
        network_output: np.ndarray,
        vehicle_number: int):
        """ generate the edge action from the neural network output.
        Args:
            network_output: the output of the neural network.
        Returns:
            the edge action.
        """
        bandwidth_allocation = np.zeros((vehicle_number,))
        bandwidth = self.rescale_the_list_to_small_than_one(list(network_output))
        for index, values in enumerate(bandwidth):
            bandwidth_allocation[index] = values * edge_node.get_bandwidth()

        edge_action = edgeAction(
            edge=edge_node,
            now_time=now_time,
            vehicle_number=vehicle_number,
            bandwidth_allocation=bandwidth_allocation,
            action_time=action_time
        )

        if not edge_action.check_action(now_time):
            raise ValueError("the edge action is invalid.")

        return edge_action
    
    def generate_weights(self, count=1, m=1) -> None:
        n = self._config.weighting_number
        all_weights = []
        target = np.random.dirichlet(np.ones(n), 1)[0]
        prev_t = target
        for _ in range(count // m):
            target = np.random.dirichlet(np.ones(n), 1)[0]
            if m == 1:
                all_weights.append(target)
            else:
                for i in range(m):
                    i_w = target * (i + 1) / float(m) + prev_t * \
                        (m - i - 1) / float(m)
                    all_weights.append(i_w)
            prev_t = target + 0.
        # print("all_weights: ", all_weights)
        for i in range(self._config.weighting_number):
            self._weights[i] = all_weights[0][i]
        # print("self._weights: ", self._weights)


Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""
    observations: NestedSpec
    vehicle_observations: NestedSpec
    vehicle_all_observations: NestedSpec
    edge_observations: NestedSpec
    actions: NestedSpec
    vehicle_actions: NestedSpec
    edge_actions: NestedSpec
    rewards: NestedSpec
    weights: NestedSpec
    critic_vehicle_actions: NestedSpec
    critic_vehicle_other_actions: NestedSpec
    critic_edge_actions: NestedSpec
    discounts: NestedSpec


def make_environment_spec(environment: vehicularNetworkEnv) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        vehicle_observations=environment.vehicle_observation_spec(),
        vehicle_all_observations=environment.vehicle_all_observation_spec(),
        edge_observations=environment.edge_observation_spec(),
        actions=environment.action_spec(),
        vehicle_actions=environment.vehicle_action_spec(),
        edge_actions=environment.edge_action_spec(),
        rewards=environment.reward_spec(),
        weights=environment.weights_spec(),
        critic_vehicle_actions=environment.vehicle_critic_network_action_spec(),
        critic_vehicle_other_actions=environment.vehicle_critic_network_other_action_spec(),
        critic_edge_actions=environment.edge_critic_network_action_spec(),
        discounts=environment.discount_spec())
    