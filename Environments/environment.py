
"""Vehicular Network Environments."""

from dm_env import specs
from acme.types import NestedSpec
import numpy as np
from Environments.dataStruct import applicationList, edge, edgeAction, informationList, informationPacket, informationRequirements, location, timeSlots, vehicleAction, vehicleList, viewList  
from typing import List, Tuple, NamedTuple
from Environments._environment import baseEnvironment, TimeStep, restart, termination, transition
import Environments.environmentConfig as env_config
from Environments.utilities import sensingAndQueuing, v2iTransmission
from Log.logger import myapp


class vehicularNetworkEnv(baseEnvironment):
    """Vehicular Network Environment built on the dm_env framework."""
    reward_history: List[List[float]] = None

    @classmethod
    def init_reward_history(cls, time_slots_number) -> None:
        """Set the reward history."""
        cls.reward_history = [[] for _ in range(time_slots_number)]

    @classmethod
    def append_reward_at_now(cls, now: int, reward: float) -> None:
        cls.reward_history[now].append(reward)

    @classmethod
    def get_reward_history_at_now(cls, now: int) -> List[float]:
        return cls.reward_history[now]

    @classmethod
    def get_min_reward_at_now(cls, now: int) -> float:
        return min(cls.reward_history[now])

    @classmethod
    def get_max_reward_at_now(cls, now: int) -> float:
        return max(cls.reward_history[now])

    def __init__(
        self, 
        envConfig: env_config.vehicularNetworkEnvConfig = None) -> None:
        """Initialize the environment."""
        if envConfig is None:
            self._config = env_config.vehicularNetworkEnvConfig()
        else:
            self._config = envConfig

        if vehicularNetworkEnv.reward_history is None:
            vehicularNetworkEnv.init_reward_history(self._config.time_slot_number)

        self._time_slots: timeSlots = timeSlots(
            start=self._config.time_slot_start,
            end=self._config.time_slot_end,
            slot_length=self._config.time_slot_length,
        )

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
        self._timeliness_views_history: List[float] = []
        self._consistency_views_history: List[float] = []
        self._redundancy_views_history: List[float] = []
        self._cost_views_history: List[float] = []

        self._reward: np.ndarray = np.zeros(self._reward_size)

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
        self._reset_next_step = False
        observation = self._observation()
        vehicle_observation = vehicularNetworkEnv.get_vehicle_observations(
            vehicle_number=self._config.vehicle_number,
            information_number=self._config.information_number,
            sensed_information_number=self._config.sensed_information_number,
            vehicle_observation_size=self._vehicle_observation_size,
            observation=observation,
            is_output_two_dimension=True,
        )
        return restart(observation=self._observation(), vehicle_observation=vehicle_observation)

    def step(self, action: np.ndarray) -> TimeStep:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """

        if self._reset_next_step:
            return self.reset()
        
        views_required_number, information_type_required_by_views_at_now, vehicle_actions, edge_action = \
            self.transform_action_array_to_actions(action)
        # myapp.dubug(f"\ntimestep:\n{self._time_slots.now()}")
        # myapp.debug(f"\naction:\n{action}")
        # myapp.debug(f"\nviews_required_number:\n{views_required_number}")
        # myapp.debug(f"\ninformation_type_required_by_views_at_now:\n{information_type_required_by_views_at_now}")
        # myapp.debug(f"\nvehicle_actions:\n{str([str(vehicle_action) for vehicle_action in vehicle_actions])}")
        # myapp.debug(f"\nedge_action:\n{edge_action}")
        
        """Compute the baseline reward and difference rewards."""
        information_objects_ordered_by_views = self.compute_information_objects(
            views_required_number=views_required_number,
            information_type_required_by_views_at_now=information_type_required_by_views_at_now,
            vehicle_actions=vehicle_actions,
            edge_action=edge_action,
        )
        # myapp.debug(f"information_objects_ordered_by_views:\n{self.string_of_information_objects_ordered_by_views(information_objects_ordered_by_views)}")

        baseline_reward = self.compute_reward(
            information_objects_ordered_by_views=information_objects_ordered_by_views,
            vehicle_actions=vehicle_actions,
        )
        # myapp.dubug(f"\nbaseline_reward:\n{baseline_reward}")

        self._reward[-1] = baseline_reward
        for i in range(self._config.vehicle_number):
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
        reward_history_at_now = vehicularNetworkEnv.get_reward_history_at_now(int(self._time_slots.now()))
        # myapp.dubug(f"\nreward_history_at_now:\n{reward_history_at_now}")
        if len(reward_history_at_now) == 0:
            edge_reward = baseline_reward
        elif len(reward_history_at_now) == 1:
            min_reward_history_at_now = vehicularNetworkEnv.get_min_reward_at_now(int(self._time_slots.now()))
            edge_reward = baseline_reward - min_reward_history_at_now
        elif len(reward_history_at_now) > 1:
            min_reward_history_at_now = vehicularNetworkEnv.get_min_reward_at_now(int(self._time_slots.now()))
            max_reward_history_at_now = vehicularNetworkEnv.get_max_reward_at_now(int(self._time_slots.now()))
            if (max_reward_history_at_now - min_reward_history_at_now) == 0:
                edge_reward = baseline_reward - min_reward_history_at_now
            else:
                edge_reward = (baseline_reward - min_reward_history_at_now) / (max_reward_history_at_now - min_reward_history_at_now)
        else:
            raise ValueError("len(reward_history_at_now) = {}".format(len(reward_history_at_now)))
        self._reward[-2] = edge_reward

        # myapp.dubug(f"\nreward:\n{self._reward}")

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

        # myapp.debug(f"\ninformation_objects_ordered_by_views:\n{self.string_of_information_objects_ordered_by_views(information_objects_ordered_by_views)}")
        
        observation = self._observation()
        vehicle_observation = vehicularNetworkEnv.get_vehicle_observations(
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
            return termination(observation=observation, reward=self._reward, vehicle_observation=vehicle_observation)
        self._time_slots.add_time()
        return transition(observation=observation, reward=self._reward, vehicle_observation=vehicle_observation)

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

        vehicle_actions: List[vehicleAction] = []
        for i in range(self._config.vehicle_number):
            vehicle_actions.append(
                vehicleAction.generate_from_np_array(
                    vehicle_index=i,
                    now_time=self._time_slots.now(),
                    vehicle_list=self._vehicle_list,
                    information_list=self._information_list,
                    sensed_information_number=self._config.sensed_information_number,
                    network_output=vhielce_action_array[i * self._vehicle_action_size: (i + 1) * self._vehicle_action_size],
                    white_gaussian_noise=self._config.white_gaussian_noise,
                    mean_channel_fading_gain=self._config.mean_channel_fading_gain,
                    second_moment_channel_fading_gain=self._config.second_moment_channel_fading_gain,
                    edge_location=self._edge_node.get_edge_location(),
                    path_loss_exponent=self._config.path_loss_exponent,
                    SNR_target_low_bound=self._config.SNR_target_low_bound,
                    SNR_target_up_bound=self._config.SNR_target_up_bound,
                    probabiliity_threshold=self._config.probabiliity_threshold,
                    action_time=self._time_slots.now(),
                )
            )
        
        edge_action: edgeAction = edgeAction.generate_from_np_array(
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
        edge_action: edgeAction,
        vehicle_index: int = -1) -> List[List[informationPacket]]:
        """Compute the reward.
        Args:
            views_required_number: the number of views required by applications.
            information_type_required_by_views_at_now: the information type required by the views.
            vehicle_actions: the actions of the vehicle.
            edge_action: the action of the edge node.
            vehicle_index: the index of the vehicle which do nothing, i.e., its action is null.
                the default value is -1, which means no vehicles do nothing.
        Returns:
            reward: the reward of the vehicle.
        """

        information_objects_ordered_by_views: List[List[informationPacket]] = []
        for i in range(views_required_number):
            information_objects_ordered_by_views.append(list())

        for i in range(self._config.vehicle_number):
            
            if i == vehicle_index:     # the vehicle do nothing
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
        vehicle_index: int = -1) -> float:

        """Compute the timeliness of views"""
        timeliness_views = []
        for information_objects in information_objects_ordered_by_views:
            timeliness_list = []
            for _ in range(self._config.vehicle_number):
                timeliness_list.append(list())
            for infor in information_objects:
                timeliness_list[infor.get_vehicle_index()].append(
                    infor.get_arrival_moment() + infor.get_queuing_time() + infor.get_transmission_time() - infor.get_updating_moment()
                )
            timeliness_of_vehicles = []
            for values in timeliness_list:
                if values == []:
                    timeliness_of_vehicles.append(0)
                else:
                    timeliness_of_vehicles.append(max(values))
            timeliness = sum(timeliness_of_vehicles)
            if not np.isinf(timeliness) and not np.isnan(timeliness):
                timeliness_views.append(timeliness)
                self._timeliness_views_history.append(timeliness)

        """Compute the consistency of views"""
        consistency_views = []
        for information_objects in information_objects_ordered_by_views:
            updating_moments_of_informations = []
            for infor in information_objects:
                updating_moments_of_informations.append(infor.get_updating_moment())
            consistency = max(updating_moments_of_informations) - min(updating_moments_of_informations)
            if not np.isinf(consistency) and not np.isnan(consistency):
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
                redundancy_list[int(infor.get_type())].append(1)
            for i in range(self._config.information_number):
                if sum(redundancy_list[i]) > 1:
                    redundancy += sum(redundancy_list[i]) - 1
            if not np.isinf(redundancy) and not np.isnan(redundancy):
                redundancy_views.append(redundancy)
                self._redundancy_views_history.append(redundancy)

        """Compute the cost of view"""
        cost_views = []
        for information_objects in information_objects_ordered_by_views:
            cost_list = []
            for _ in range(self._config.vehicle_number):
                cost_list.append(list())
            for infor in information_objects:
                if infor.get_vehicle_index() != -1:
                    cost_list[infor.get_vehicle_index()].append(
                        self._vehicle_list.get_vehicle(infor.get_vehicle_index()).get_sensing_cost_by_type(infor.get_type()) + \
                            infor.get_transmission_time() * vehicle_actions[infor.get_vehicle_index()].get_transmission_power()
                    )
            cost_of_vehicles = []
            for values in cost_list:
                if values == []:
                    cost_of_vehicles.append(0)
                else:
                    cost_of_vehicles.append(sum(values))
            cost = sum(cost_of_vehicles)
            if not np.isinf(cost) and not np.isnan(cost):
                cost_views.append(cost)
                self._cost_views_history.append(cost)

        """Normalize the timeliness, consistency, redundancy, and cost of views"""
        timeliness_views_normalized = []
        consistency_views_normalized = []
        redundancy_views_normalized = []
        cost_views_normalized = []

        for i in range(len(timeliness_views)):
            if (max(self._timeliness_views_history) - min(self._timeliness_views_history)) != 0 and \
                not np.isnan(timeliness_views[i] - min(self._timeliness_views_history)) / (max(self._timeliness_views_history) - min(self._timeliness_views_history)):
                # myapp.dubug("(timeliness_views[i] - min(self._timeliness_views_history)) / (max(self._timeliness_views_history) - min(self._timeliness_views_history)):")
                # myapp.dubug((timeliness_views[i] - min(self._timeliness_views_history)) / (max(self._timeliness_views_history) - min(self._timeliness_views_history)))
                # myapp.dubug(timeliness_views[i])
                # myapp.dubug(min(self._timeliness_views_history))
                # myapp.dubug(max(self._timeliness_views_history))
                # myapp.dubug("\n")
                timeliness_views_normalized.append(
                    (timeliness_views[i] - min(self._timeliness_views_history)) / (max(self._timeliness_views_history) - min(self._timeliness_views_history))
                )
            else:
                timeliness_views_normalized.append(-1)
            
            if (max(self._consistency_views_history) - min(self._consistency_views_history)) != 0 and \
                not np.isnan(consistency_views[i] - min(self._consistency_views_history)) / (max(self._consistency_views_history) - min(self._consistency_views_history)):
                # myapp.dubug("(consistency_views[i] - min(self._consistency_views_history)) / (max(self._consistency_views_history) - min(self._consistency_views_history)):")
                # myapp.dubug((consistency_views[i] - min(self._consistency_views_history)) / (max(self._consistency_views_history) - min(self._consistency_views_history)))
                # myapp.dubug(consistency_views[i])
                # myapp.dubug(min(self._consistency_views_history))
                # myapp.dubug(max(self._consistency_views_history))
                # myapp.dubug("\n")
                consistency_views_normalized.append(
                    (consistency_views[i] - min(self._consistency_views_history)) / (max(self._consistency_views_history) - min(self._consistency_views_history))
                )
            else:
                consistency_views_normalized.append(-1)
            
            if (max(self._redundancy_views_history) - min(self._redundancy_views_history)) != 0 and \
                not np.isnan((redundancy_views[i] - min(self._redundancy_views_history)) / (max(self._redundancy_views_history) - min(self._redundancy_views_history))):
                # myapp.dubug("(redundancy_views[i] - min(self._redundancy_views_history)) / (max(self._redundancy_views_history) - min(self._redundancy_views_history)):")
                # myapp.dubug((redundancy_views[i] - min(self._redundancy_views_history)) / (max(self._redundancy_views_history) - min(self._redundancy_views_history)))
                # myapp.dubug(redundancy_views[i])
                # myapp.dubug(min(self._redundancy_views_history))
                # myapp.dubug(max(self._redundancy_views_history))
                # myapp.dubug("\n")
                redundancy_views_normalized.append(
                    (redundancy_views[i] - min(self._redundancy_views_history)) / (max(self._redundancy_views_history) - min(self._redundancy_views_history))
                )
            else:
                redundancy_views_normalized.append(-1)

            if (max(self._cost_views_history) - min(self._cost_views_history)) != 0 and \
                not np.isnan(cost_views[i] - min(self._cost_views_history)) / (max(self._cost_views_history) - min(self._cost_views_history)):
                # myapp.dubug("(cost_views[i] - min(self._cost_views_history)) / (max(self._cost_views_history) - min(self._cost_views_history)):")
                # myapp.dubug((cost_views[i] - min(self._cost_views_history)) / (max(self._cost_views_history) - min(self._cost_views_history)))
                # myapp.dubug(cost_views[i])
                # myapp.dubug(min(self._cost_views_history))
                # myapp.dubug(max(self._cost_views_history))
                # myapp.dubug("\n")
                cost_views_normalized.append(
                    (cost_views[i] - min(self._cost_views_history)) / (max(self._cost_views_history) - min(self._cost_views_history))
                )
            else:
                cost_views_normalized.append(-1)

        """Compute the age of view."""
        age_of_view = []
        for i in range(len(timeliness_views_normalized)):
            if timeliness_views_normalized[i] != -1 and consistency_views_normalized[i] != -1 and redundancy_views_normalized[i] != -1 and cost_views_normalized[i] != -1:
                age_of_view.append(
                    self._config.wight_of_timeliness * timeliness_views_normalized[i] + \
                    self._config.wight_of_consistency * consistency_views_normalized[i] + \
                    self._config.wight_of_redundancy * redundancy_views_normalized[i] + \
                    self._config.wight_of_cost * cost_views_normalized[i]
                )

        if len(age_of_view) == 0:
            return -1

        reward = float(1.0 - sum(age_of_view) / len(age_of_view))
        reward = 0 if reward < 0 else reward
        reward = 1 if reward > 1 else reward

        if vehicle_index == -1:
            vehicularNetworkEnv.append_reward_at_now(
                now=int(self._time_slots.now()),
                reward=reward,
            )

        return reward

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
        return specs.Array(
            shape=(self._reward_size,), 
            dtype=float, 
            name='rewards'
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
    critic_vehicle_actions: NestedSpec
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
        critic_vehicle_actions=environment.vehicle_critic_network_action_spec(),
        critic_edge_actions=environment.edge_critic_network_action_spec(),
        discounts=environment.discount_spec())