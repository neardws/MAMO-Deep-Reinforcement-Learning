import numpy as np
import pandas as pd

from Environments.utilities import v2iTransmission
# from utilities import v2iTransmission

class timeSlots(object):
    """The set of discrete time slots of the system"""
    def __init__(
        self, 
        start: int, 
        end: int, 
        slot_length: int) -> None:
        """method to initialize the time slots
        Args:
            start: the start time of the system
            end: the end time of the system
            slot_length: the length of each time slot"""   
        self.start = start
        self.end = end
        self.now = self.start
        self.slot_length = slot_length
        self._number = int((end - start + 1) / slot_length)
    
    def add_time(self) -> None:
        """method to add time to the system"""
        self.now += 1

    def is_end(self) -> bool:
        """method to check if the system is at the end of the time slots"""
        return self.now > self.end

    def get_slot_length(self):
        """method to get the length of each time slot"""
        return self.slot_length

    def get_number(self) -> int:
        return self._number

    def now(self) -> int:
        return self.now

    def reset(self) -> None:
        self.now = self.start


class information(object):
    """
    the object of information, which is used to store the information flow,
    including information generation, queuing, transmission, 
    and finally received at the edge.
    """
    def __init__(
        self,
        type: int,
        vehicle_no: int,
        edge_no: int = -1,
        updating_moment: float = -1,
        inter_arrival_interval: float = -1,
        arrival_moment: float = -1,
        queuing_time: float = -1,
        transmission_time: float = -1,
        received_moment: float = -1) -> None:
        """ initialize the information.
        Args:
            type: the type of the information.
            vehicle_no: the index of the vehicle.
            edge_no: the index of the edge.
            updating_moment: the generation time of the information.
            inter_arrival_interval: the inter-arrival interval of the information.
            arrival_moment: the arrival moment of the information.
            queuing_time: the queuing time of the information.
            transmission_time: the transmission time of the information.
            received_moment: the received moment of the information.
        """
        self._type = type
        self._vehicle_no = vehicle_no
        self._edge_no = edge_no
        self._updating_moment = updating_moment
        self._inter_arrival_interval = inter_arrival_interval
        self._arrival_moment = arrival_moment
        self._queuing_time = queuing_time
        self._transmission_time = transmission_time
        self._received_moment = received_moment

    def get_type(self) -> int:
        return self._type
    
    def set_type(self, type: int) -> None:
        self._type = type
    
    def get_vehicle_no(self) -> int:
        return self._vehicle_no
    
    def set_vehicle_no(self, vehicle_no: int) -> None:
        self._vehicle_no = vehicle_no
    
    def get_edge_no(self) -> int:
        return self._edge_no

    def set_edge_no(self, edge_no: int) -> None:
        self._edge_no = edge_no
    
    def get_updating_moment(self) -> float:
        return self._updating_moment
    
    def set_updating_moments(self, updating_moment: float) -> None:
        self._updating_moment = updating_moment

    def get_inter_arrival_interval(self) -> float:
        return self._inter_arrival_interval

    def set_inter_arrival_interval(self, inter_arrival_interval: float) -> None:
        self._inter_arrival_interval = inter_arrival_interval

    def get_arrival_moment(self) -> float:
        return self._arrival_moment

    def set_arrrival_moment(self, arrival_moment: float) -> None:
        self._arrival_moment = arrival_moment
    
    def get_queuing_time(self) -> float:
        return self._queuing_time
    
    def set_queuing_time(self, queuing_time: float) -> None:
        self._queuing_time = queuing_time
    
    def get_transmission_time(self) -> float:
        return self._transmission_time
    
    def set_transmission_time(self, transmission_time: float) -> None:
        self._transmission_time = transmission_time
    
    def get_received_moment(self) -> float:
        return self._received_moment
    
    def set_received_moment(self, received_moment: float) -> None:
        self._received_moment = received_moment
    

class location(object):
    """ the location of the node. """
    def __init__(self, x: float, y: float) -> None:
        """ initialize the location.
        Args:
            x: the x coordinate.
            y: the y coordinate.
        """
        self._x = x
        self._y = y

    def get_x(self) -> float:
        return self._x

    def get_y(self) -> float:
        return self._y

    def get_distance(self, location: "location") -> float:
        """ get the distance between two locations.
        Args:
            location: the location.
        Returns:
            the distance.
        """
        return np.math.sqrt(
            (self._x - location.get_x())**2 + 
            (self._y - location.get_y())**2
        )

class trajectory(object):
    """ the trajectory of the node. """
    def __init__(self, timeSlots: timeSlots, locations: list) -> None:
        """ initialize the trajectory.
        Args:
            max_time_slots: the maximum number of time slots.
            locations: the location list.
        """
        self._max_time_slots = timeSlots.get_number()
        self._locations = locations

        if len(self._locations) != self._max_timestampes:
            raise ValueError("The number of locations must be equal to the max_timestampes.")

    def get_location(self, nowTimeSlot: int):
        """ get the location of the timestamp.
        Args:
            timestamp: the timestamp.
        Returns:
            the location.
        """
        return self._locations[nowTimeSlot]

    def get_locations(self) -> list:
        """ get the locations.
        Returns:
            the locations.
        """
        return self._locations


class vehicle(object):
    """" the vehicle. """
    def __init__(
        self, 
        vehicle_no: int,
        vehicle_trajectory: trajectory,
        information_number: int,
        sened_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        seed: int) -> None:
        """ initialize the vehicle.
        Args:
            vehicle_no: the index of vehicle. e.g. 0, 1, 2, ...
            vehicle_trajectory: the trajectory of the vehicle.
            information_number: the number of information list.
            sened_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seed: the random seed.
        """
        self._vehicle_no = vehicle_no
        self._vehicle_trajectory = vehicle_trajectory
        self._information_number = information_number
        self._sensed_information_number = sened_information_number
        self._min_sensing_cost = min_sensing_cost
        self._max_sensing_cost = max_sensing_cost
        self._transmission_power = transmission_power
        self._seed = seed

        if self._sensed_information_number > self._information_number:
            raise ValueError("The max information number must be less than the information number.")
        
        self._information_canbe_sensed = self.information_types_can_be_sensed()

        self._sensing_cost = self.sensing_cost_of_information()

    def get_vehicle_no(self) -> int:
        return self._vehicle_no

    def get_transmission_power(self) -> float:
        return self._transmission_power

    def information_types_can_be_sensed(self) -> list:
        np.random.seed(self._seed)
        return list(np.random.choice(
            a=self._information_number,
            size=self._sensed_information_number,
            replace=False))

    def sensing_cost_of_information(self) -> list:
        np.random.seed(self._seed)
        return list(np.random.uniform(
            low=self._min_sensing_cost,
            high=self._max_sensing_cost,
            size=self._sensed_information_number
        ))

    def get_sensing_cost(self) -> list:
        return self._sensing_cost
    
    def get_sensing_cost_by_type(self, type: int) -> float:
        for _ in range(self._sensed_information_number):
            if self._information_canbe_sensed[_] == type:
                return self._sensing_cost[_]
    
    def get_vehicle_location(self, nowTimeSlot: int) -> location:
        return self._vehicle_trajectory.get_location(nowTimeSlot)

    def get_distance_between_edge(self, nowTimeSlot: int, edge_location) -> float:
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)

    def get_sensed_information_number(self) -> int:
        return self._sensed_information_number
    
    def get_information_canbe_sensed(self) -> list:
        return self._information_canbe_sensed
    
    def get_vehicle_trajectory(self) -> trajectory:
        return self._vehicle_trajectory

    def set_information_number(self, information_number: int) -> None:
        self._information_number = information_number
    
    def set_sensed_information_number(self, sensed_information_number: int) -> None:
        self._sensed_information_number = sensed_information_number

    def set_transmission_power(self, transmission_power: float) -> None:
        self._transmission_power = transmission_power
    
    def set_min_sensing_cost(self, min_sensing_cost: float) -> None:
        self._min_sensing_cost = min_sensing_cost
    
    def set_max_sensing_cost(self, max_sensing_cost: float) -> None:
        self._max_sensing_cost = max_sensing_cost

class vehicleList(object):
    """ the vehicle list. """
    def __init__(
        self, 
        number: int, 
        trajectories_file_name: str,
        information_number: int,
        sensed_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        seeds: list) -> None:
        """ initialize the vehicle list.
        Args:
            number: the number of vehicles.
            trajectories_file_name: the file name of the vehicle trajectories.
            information_number: the number of information list.
            sensed_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seeds: the random seed list.
        """
        self._number = number
        self._trajectories_file_name = trajectories_file_name
        self._information_number = information_number
        self._sensed_information_number = sensed_information_number
        self._min_sensing_cost = min_sensing_cost
        self._max_sensing_cost = max_sensing_cost
        self._transmission_power = transmission_power
        self._seeds = seeds

        self._vehicle_trajectories = self.read_vehicle_trajectories()

        self._vehicle_list = []
        for i in range(self._vehicles_number):
            self._vehicle_list.append(
                vehicle(
                    vehicle_no=i,
                    vehicle_trajectory=self._vehicle_trajectories[i],
                    information_number=self._information_number,
                    sensed_information_number=self._sensed_information_number,
                    min_sensing_cost=self._min_sensing_cost,
                    max_sensing_cost=self._max_sensing_cost,
                    transmission_power=self._transmission_power,
                    seed=self._seeds[i]
                )
            )

    def get_vehicle_list(self) -> list:
        return self._vehicle_list

    def get_number(self) -> int:
        return self._number

    def get_sensed_information_number(self) -> int:
        return self._sensed_information_number

    def get_vehicle(self, vehicle_no: int) -> vehicle:
        return self._vehicle_list[vehicle_no]

    def read_vehicle_trajectories(self, timeSlots: timeSlots) -> list:

        df = pd.read_csv(
            self._vehicle_trajectories_file_name, 
            names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)

        max_vehicle_id = df['vehicle_id'].max()

        selected_vehicle_id = []
        for vehicle_id in range(int(max_vehicle_id)):
            new_df = df[df['vehicle_id'] == vehicle_id]
            max_x = new_df['longitude'].max()
            max_y = new_df['latitude'].max()
            min_x = new_df['longitude'].min()
            min_y = new_df['latitude'].min()
            distance = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
            selected_vehicle_id.append(
                {
                    "vehicle_id": vehicle_id,
                    "distance": distance
                })

        selected_vehicle_id.sort(key=lambda x : x["distance"], reverse=True)
        new_vehicle_id = 0
        vehicle_trajectories = []
        for vehicle_id in selected_vehicle_id[ : self._vehicles_number]:
            new_df = df[df['vehicle_id'] == vehicle_id["vehicle_id"]]
            loc_list = []
            for row in new_df.itertuples():
                # time = getattr(row, 'time')
                x = getattr(row, 'longitude')
                y = getattr(row, 'latitude')
                loc = location(x, y)
                loc_list.append(loc)
            new_vehicle_trajectory = trajectory(
                vehicle_id=new_vehicle_id,
                timeSlots=timeSlots,
                loc_list=loc_list
            )
            new_vehicle_id += 1
            vehicle_trajectories.append(new_vehicle_trajectory)

        return vehicle_trajectories


class vehicleAction(object):
    """ the action of the vehicle. """
    def __init__(
        self, 
        vehicle_no: int,
        now_time: int,
        sensed_information: list = None,
        sensing_frequencies: list = None,
        uploading_priorities: list = None,
        transmission_power: float = None, 
        action_time: int = None) -> None:
        """ initialize the vehicle action.
        Args:
            vehicle_no: the index of vehicle. e.g. 0, 1, 2, ...
            now_time: the current time.
            vehicle_list: the vehicle list.
            sensed_information: the sensed information.
                e.g., 0 or 1, indicates whether the information is sensed or not.
                and the type of information is rocorded in vehicle.information_canbe_sensed .
            sensing_frequencies: the sensing frequencies.
            uploading_priorities: the uploading priorities.
            transmission_power: the transmission power.
            action_time: the time of the action.
        """
        self._vehicle_no = vehicle_no
        self._now_time = now_time
        self._sensed_information = sensed_information
        self._sensing_frequencies = sensing_frequencies
        self._uploading_priorities = uploading_priorities
        self._transmission_power = transmission_power
        self._action_time = action_time

        # if not self.check_action(now_time, vehicle_list):
        #     raise ValueError("The action is not valid.")

    def check_action(self, nowTimeSlot: int, vehicleList: vehicleList) -> bool:
        """ check the action.
        Args:
            nowTimeSlot: the time of the action.
        Returns:
            True if the action is valid.
        """
        if self._action_time != nowTimeSlot:
            return False
        if self._vehicle_no >= len(vehicleList.get_vehicle_list()):
            return False
        vehicle = vehicleList.get_vehicle(self._vehicle_no)
        if not (len(self._sensed_information) == len(self._sensing_frequencies) \
            == len(self._uploading_priorities) == len(vehicle.get_sensed_information_number())):
            return False
        if self._transmission_power > vehicle.get_transmission_power():
            return False
        return True

    def get_sensed_information(self) -> list:
        return self._sensed_information

    def get_sensing_frequencies(self) -> list:
        return self._sensing_frequencies
    
    def get_uploading_priorities(self) -> list:
        return self._uploading_priorities

    def get_transmission_power(self) -> float:
        return self._transmission_power

    def get_action_time(self) -> int:
        return self._action_time

    @staticmethod
    def generate_from_np_array(
        now_time: int,
        vehicle_no: int,
        vehicle_list: vehicleList,
        max_information_number: int,
        network_output: np.array,
        white_gaussian_noise: int,
        mean_channel_fading_gain: float,
        second_moment_channel_fadding_gain: float,
        edge_location: location,
        path_loss_exponent: int,
        SNR_target: float,
        probabiliity_threshold: float,
        action_time: int):
        """ generate the vehicle action from the neural network output.

        self._vehicle_action_size = self._max_information_number + self._max_information_number + \
            self._max_information_number + 1
            # sensed_information + sensing_frequencies + uploading_priorities + transmission_power

        Args:
            network_output: the output of the neural network.
        Returns:
            the vehicle action.
        """
        sensed_information = np.zeros(shape=(max_information_number,))
        sensing_frequencies = np.zeros(shape=(max_information_number,))
        uploading_priorities = np.zeros(shape=(max_information_number,))

        for index, values in enumerate(network_output[:max_information_number]):
            if values > 0.5:
                sensed_information[index] = 1
        for index, values in enumerate(network_output[max_information_number: 2*max_information_number]):
            if sensed_information[index] == 1:
                sensing_frequencies[index] = values
        for index, values in enumerate(network_output[2*max_information_number: 3*max_information_number]):
            if sensed_information[index] == 1:
                uploading_priorities[index] = values

        sensed_information = list(sensed_information)
        sensing_frequencies = list(sensing_frequencies)
        uploading_priorities = list(uploading_priorities)

        minimum_transmission_power = v2iTransmission.get_minimum_transmission_power(
            white_gaussian_noise=white_gaussian_noise,
            mean_channel_fading_gain=mean_channel_fading_gain,
            second_moment_channel_fadding_gain=second_moment_channel_fadding_gain,
            distance=vehicle_list.get_vehicle(vehicle_no).get_vehicle_location(now_time).get_distance(edge_location),
            path_loss_exponent=path_loss_exponent,
            transmission_power=vehicle_list.get_vehicle(vehicle_no).get_transmission_power(),
            SNR_target=SNR_target,
            probabiliity_threshold=probabiliity_threshold
        )

        transmisson_power = minimum_transmission_power + network_output[-1] * \
            (vehicle_list.get_vehicle(vehicle_no).get_transmission_power() - minimum_transmission_power)
        
        vehicle_action = vehicleAction(
            vehicle_no=vehicle_no,
            now_time=now_time,

            sensed_information_types=sensed_information,
            sensing_frequencies=sensing_frequencies,
            uploading_priorities=uploading_priorities,
            transmission_power=transmisson_power,

            action_time=action_time,
        )

        if not vehicle_action.check_action(now_time, vehicle_list):
            raise ValueError("The vehicle action is not valid.")

        return vehicle_action


class edge(object):
    """ the edge. """
    def __init__(
        self, 
        edge_no: int,
        information_number: int,
        edge_location: location,
        communication_range: float,
        bandwidth: float) -> None:
        """ initialize the edge.
        Args:
            edge_no: the index of edge. e.g. 0, 1, 2, ...
            information_number: the number of information list.
            edge_location: the location of the edge.
            communication_range: the range of V2I communications.
            bandwidth: the bandwidth of edge.
        """
        self._edge_no = edge_no
        self._information_number = information_number
        self._edge_location = edge_location
        self._communication_range = communication_range
        self._bandwidth = bandwidth
        self._information_in_edge = np.zeros(shape=(self._information_number,), dtype=np.int)

    def get_edge_no(self) -> int:
        return self._edge_no

    def get_edge_location(self) -> location:
        return self._edge_location

    def get_communication_range(self) -> float:
        return self._communication_range
    
    def get_bandwidth(self) -> float:
        return self._bandwidth

    def set_edge_location(self, edge_location: location) -> None:
        self._edge_location = edge_location

    def set_communication_range(self, communication_range: float) -> None:
        self._communication_range = communication_range
    
    def set_bandwidth(self, bandwidth: float) -> None:
        self._bandwidth = bandwidth

class edgeAction(object):
    """ the action of the edge. """
    def __init__(
        self, 
        edge: edge,
        now_time: int,
        vehicle_number: int,
        bandwidth_allocation: np.array,
        action_time: int) -> None:
        """ initialize the edge action.
        Args:
            edge: the edge.
            now_time: the current time.
            vehicle_number: the number of vehicles.
            action_time: the time of the action.
        """
        self._edge_bandwidth = edge.get_bandwidth()
        self._now_time = now_time
        self._vehicle_number = vehicle_number
        self._action_time = action_time
        self._bandwidth_allocation = bandwidth_allocation

    def get_bandwidth_allocation(self) -> np.array:
        return self._bandwidth_allocation

    def get_the_sum_of_bandwidth_allocation(self) -> float:
        return np.sum(self._bandwidth_allocation)

    def check_action(self, nowTimeSlot: int) -> bool:
        """ check the action.
        Args:
            nowTimeSlot: the time of the action.
        Returns:
            True if the action is valid.
        """
        if self._action_time != nowTimeSlot:
            return False
        if self._vehicle_number != len(self._bandwidth_allocation):
            return False
        if self._edge_bandwidth < self.get_the_sum_of_bandwidth_allocation():
            return False
        return True

    @staticmethod
    def generate_from_np_array(
        now_time: int,
        edge_node: edge,
        action_time: int,
        network_output: np.array,
        vehicle_number: int):
        """ generate the edge action from the neural network output.
        Args:
            network_output: the output of the neural network.
        Returns:
            the edge action.
        """
        bandwidth_allocation = np.zeros((vehicle_number,))
        for index, values in enumerate(network_output):
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


class applicationList(object):
    """
    This class is used to store the application list of the environment.
    """
    def __init__(
        self, 
        number: int,
        view_number: int,
        views_per_application: int,
        seed: int) -> None:
        """ initialize the application list.
        Args:
            number: the number of application list.
            view_number: the number of view list.
            seed: the random seed.
        """
        self._number = number
        self._view_number = view_number
        self._views_per_application = views_per_application
        self._seed = seed
        self._application_list = []
        
        if self._views_per_application == 1:
            if self._number != self._view_number:
                self._number = self._view_number
            np.random.seed(self._seed)
            self.application_list = list(np.random.permutation(list(range(self._number))))
        elif self._views_per_application > 1:
            # TODO: to generate the mapping between application and view list.
            pass
        else:
            raise Exception("The views_per_application must be greater than 1.")

    def get_number(self) -> int:
        return self._number

    def get_application_list(self) -> list:
        if self.application_list is None:
            raise Exception("The application list is not list.")
        return self.application_list

    def set_application_list(self, application_list) -> None:
        self.application_list = application_list


class viewList(object):
    """ the view list. """
    def __init__(
        self, 
        number: int, 
        information_number: int, 
        required_information_number: int, 
        seeds: list) -> None:
        """ initialize the view list.
        Args:
            number: the number of view list.
            information_number: the number of information.
            required_information_number: the maximume number of information required by one view.
            seeds: the random seeds.
        """
        self._number = number
        self._information_number = information_number
        self._required_information_number = required_information_number
        self._seeds = seeds

        if self._required_information_number > self._information_number:
            raise ValueError("The max_information_number must be less than the information_number.")

        if len(self._seeds) != self._number:
            raise ValueError("The number of seeds must be equal to the number of view lists.")

        self._view_list = []

        np.random.seed(self._seeds[0])
        self._random_information_number = np.random.randint(
            size=self._number,
            low=1,
            high=self._required_information_number
        )

        for _ in range(self._number):
            random_information_number = self._random_information_number[_]
            np.random.seed(self._seeds[_])
            self._view_list.append(
                list(np.random.choice(
                    a=self._information_number, 
                    size=random_information_number,
                    replace=False
                ))
            )

    def get_number(self) -> int:
        return self._number
        
    def get_view_list(self) -> list:
        """ get the view list.
        Returns:
            the view list.
        """
        return self._view_list



class informationList(object):
    """
    This class is used to store the information list of the environment.
    to store the whole information list, 
    which randomly initialize the characteristics of each information,
    including the type, data size, update interval.
    """
    def __init__(
        self, 
        number: int, 
        seed: int, 
        data_size_low_bound: float,
        data_size_up_bound: float,
        data_types_number: int,
        update_interval_low_bound: int,
        update_interval_up_bound: int,
        vehicle_list: vehicleList,
        edge_node: edge,
        additive_white_gaussian_noise,
        mean_channel_fadding_gain,
        second_channel_fadding_gain,
        path_loss_exponent) -> None:
        """ initialize the information list.
        Args:
            number: the number of information int the list.
            seed: the random seed.
            data_size_low_bound: the low bound of the data size.
            data_size_up_bound: the up bound of the data size.
            data_types_number: the number of data types.
            update_interval_low_bound: the low bound of the update interval.
            update_interval_up_bound: the up bound of the update interval.
        """
        self._number = number
        self._seed = seed
        self._data_size_low_bound = data_size_low_bound
        self._data_size_up_bound = data_size_up_bound
        self._data_types_number = data_types_number
        self._update_interval_low_bound = update_interval_low_bound
        self._update_interval_up_bound = update_interval_up_bound

        if self._data_types_number != self._number:
            self._data_types_number = self._number
        np.random.seed(self._seed)
        self.types_of_information = np.random.permutation(list(range(self._data_types_number)))

        np.random.seed(self._seed)
        self.data_size_of_information = np.random.uniform(
            low=self._data_size_low_bound,
            high=self._data_size_up_bound,
            size=self._number
        )

        np.random.seed(self._seed)
        self.update_interval_of_information = np.random.randint(
            size=self._number, 
            low=self._update_interval_low_bound, 
            high=self._update_interval_up_bound
        )

        self.information_list = []
        for i in range(self._number):
            self.information_list.append({
                "type": self.types_of_information[i],
                "data_size": self.data_size_of_information[i],
                "update_interval": self.update_interval_of_information[i]
            })
        
        self.mean_service_time_of_types, self.second_moment_service_time_of_types = \
            self.compute_mean_and_second_moment_service_time_of_types(
                vehicle_list=vehicle_list,
                edge_node=edge_node,
                additive_white_gaussian_noise=additive_white_gaussian_noise,
                mean_channel_fadding_gain=mean_channel_fadding_gain,
                second_channel_fadding_gain=second_channel_fadding_gain,
                path_loss_exponent=path_loss_exponent
            )

    def get_number(self) -> int:
        """ get the number of information.
        Returns:
            the number of information.
        """
        return self._number

    def get_information_list(self) -> list:
        return self.information_list
    
    def set_information_list(self, information_list) -> None:
        self.information_list = information_list
    
    def get_information_type_by_index(self, index: int) -> int:
        if index >= self._number:
            raise ValueError("The index is out of range.")
        return self.information_list[index]["type"]

    def get_information_by_type(self, type: int) -> dict:
        """method to get the information by type"""
        for information in self.information_list:
            if information["type"] == type:
                return information
        return None

    def get_information_siez_by_type(self, type: int) -> float:
        """method to get the information size by type"""
        for information in self.information_list:
            if information["type"] == type:
                return information["data_size"]
        return None

    def get_information_update_interval_by_type(self, type: int) -> int:
        """method to get the information update interval by type"""
        for information in self.information_list:
            if information["type"] == type:
                return information["update_interval"]
        return None

    def get_mean_service_time_of_types(self) -> np.array:
        return self.mean_service_time_of_types
    
    def get_second_moment_service_time_of_types(self) -> np.array:
        return self.second_moment_service_time_of_types

    def get_mean_service_time_by_vehicle_and_type(self, vehicle_no, type):
        return self.mean_service_time_of_types[vehicle_no][type]

    def get_second_moment_service_time_by_vehicle_and_type(self, vehicle_no, type):
        return self.second_moment_service_time_of_types[vehicle_no][type]

    def compute_mean_and_second_moment_service_time_of_types(
        self, 
        vehicle_list: vehicleList,
        edge_node: edge,
        additive_white_gaussian_noise,
        mean_channel_fadding_gain,
        second_channel_fadding_gain,
        path_loss_exponent):
        """
        method to get the mean and second moment service time of 
        each type of information at each vehile.
        Args:
            vehicle_list: the vehicle list.
            edge_node: the edge node.
            additive_white_gaussian_noise: the additive white gaussian noise.
            mean_channel_fadding_gain: the mean channel fadding gain.
            second_channel_fadding_gain: the second channel fadding gain.
            path_loss_exponent: the path loss exponent.
        Returns:
            the mean and second moment service time of each type of information.
        """
        vehicle_number = vehicle_list.get_vehicles_number()
        mean_service_time_of_types = np.zeros(shape=(vehicle_number, self._data_types_number))
        second_moment_service_time_of_types = np.zeros(shape=(vehicle_number, self._data_types_number))

        white_gaussian_noise = v2iTransmission.cover_dBm_to_W(additive_white_gaussian_noise)

        for vehicle_index in range(vehicle_number):
            vehicle = vehicle_list.get_vehicle(vehicle_index)
            for data_type_index in range(self._data_types_number):
                transmission_time = []
                for location in vehicle.get_vehicle_trajectory().get_locations():
                    distance = location.get_distance(edge_node.get_edge_location())
                    channel_fading_gain = v2iTransmission.generate_channel_fading_gain(
                        mean=mean_channel_fadding_gain,
                        second_moment=second_channel_fadding_gain
                    )
                    SNR = v2iTransmission.compute_SNR(
                        white_gaussian_noise=white_gaussian_noise,
                        channel_fading_gain=channel_fading_gain,
                        distance=distance,
                        path_loss_exponent=path_loss_exponent,
                        transmission_power=vehicle.get_transmission_power()
                    )
                    bandwidth = edge_node.get_bandwidth() / vehicle_number
                    transmission_time.append(self.get_information_siez_by_type(data_type_index) / v2iTransmission.compute_transmission_rate(SNR, bandwidth))
                
                mean_service_time = np.array(transmission_time).mean()
                second_moment_service_time = np.array(transmission_time).var()
                mean_service_time_of_types[vehicle_index][data_type_index] = mean_service_time
                second_moment_service_time_of_types[vehicle_index][data_type_index] = second_moment_service_time

        return mean_service_time_of_types, second_moment_service_time_of_types


class informationRequirements(object):
    """
    This class is used to store the data requirements of the environment.
    """
    def __init__(
        self,
        time_slots: timeSlots,
        max_application_number: int,
        min_application_number: int,
        application: applicationList,
        view: viewList,
        information: informationList,
        seed: int
        ) -> None:
        """ initialize the information set.
        Args:
            time_slots: the time slots.
            max_application_number: the maximum application number at each timestampe.
            min_application_number: the minimum application number at each timestampe.
            application: the application list.
            view: the view list.
            information: the information set.
            seed: the random seed.
        """
        self._max_timestampes = time_slots.get_number()     #  max_timestampes: the maximum timestamp.
        self._max_application_number = max_application_number
        self._min_application_number = min_application_number
        self._application_number = application.get_number()
        self._application_list = application.get_application_list()
        self._view_list = view.get_view_list()
        self._information_list = information.get_information_list()
        self._seed = seed

        self.applications_at_time = self.applications_at_times()
        

    def get_seed(self) -> int:
        """ get the random seed.
        Returns:
            the random seed.
        """
        return self._seed
    
    def applications_at_times(self) -> list:
        """ get the applications at each time.
        Returns:
            the applications at times.
        """
        random_application_number = np.random.randint(
            low=self._min_application_number, 
            high=self._max_application_number, 
            size=self._max_timestampes
        )

        applications_at_times = []
        for _ in range(self._max_timestampes):
            applications_at_times.append(
                list(np.random.choice(
                    list(range(self._application_number)), 
                    random_application_number[_], 
                    replace=False))
            )

        return applications_at_times
    
    def applications_at_now(self, nowTimeStamp: int) -> list:
        """ get the applications now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the applications list.
        """
        if self.applications_at_time is None:
            return Exception("applications_at_time is None.")
        return self.applications_at_time[nowTimeStamp]

    def views_required_by_application_at_now(self, nowTimeStamp: int) -> list:
        """ get the views required by application now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the views required by application list.
        """
        applications_at_now = self.applications_at_now(nowTimeStamp)
        views_required_by_application_at_now = []
        for _ in applications_at_now:
            views_required_by_application_at_now.append(
                self._application_list[_]
            )
        return views_required_by_application_at_now
    
    def information_required_by_views_at_now(self, nowTimeStamp: int) -> dict:
        """ get the information required by views now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the information set required by views list.
        """
        views_required_by_application_at_now = self.views_required_by_application_at_now(nowTimeStamp)
        views_required_number = len(views_required_by_application_at_now)
        
        information_type_required_by_views_at_now = []

        for _ in views_required_by_application_at_now:
            information_type_required = []
            view_list = self._view_list[_]
            for __ in view_list:
                information_type_required.append(
                    self._information_list[__]["type"]
                ) 
            information_type_required_by_views_at_now.append(
                information_type_required
            )

        return {
            "views_required_number": views_required_number,
            "information_type_required_by_views_at_now": information_type_required_by_views_at_now
        }
    
    def information_required_at_now(self, nowTimeStamp: int) -> np.array:
        """ get the information required now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the information set required.
        """
        views_required_by_application_at_now = self.views_required_by_application_at_now(nowTimeStamp)
        
        information_type_required_at_now = np.zeros(self._information_list.get_number())

        for view_index in views_required_by_application_at_now:
            view = self._view_list[view_index]
            for information_index in view:
                information_type_required_at_now[self._information_list[information_index]["type"]] = 1 

        return information_type_required_at_now


if __name__ == "__main__":
    application = applicationList(
        application_number=10,
        view_number=10,
        views_per_application=1,
        seed=1
    )
    print("application:\n", application.get_application_list())

    information_list = informationList(
        information_number=10, 
        seed=0, 
        data_size_low_bound=0, 
        data_size_up_bound=1, 
        data_types_number=3, 
        update_interval_low_bound=1, 
        update_interval_up_bound=3
    )
    print("information_list:\n", information_list.get_information_list())

    view = viewList(
        view_number=10,
        information_number=10,
        max_information_number=3,
        seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    print("views:\n", view.get_view_list())

    information_required = informationRequirements(
        max_timestampes=10,
        max_application_number=10,
        min_application_number=1,
        application=application,
        view=view,
        information=information_list,
        seed=0
    )
    print(information_required.applications_at_time)
    nowTimeStamp = 0
    print(information_required.applications_at_now(nowTimeStamp))
    print(information_required.views_required_by_application_at_now(nowTimeStamp))
    print(information_required.information_required_by_views_at_now(nowTimeStamp))

    """Print the result."""
    """
    application:
    [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]

    information_list:
    [{'type': 2, 'data_size': 0.5488135039273248, 'update_interval': 1}, 
    {'type': 8, 'data_size': 0.7151893663724195, 'update_interval': 2}, 
    {'type': 4, 'data_size': 0.6027633760716439, 'update_interval': 2}, 
    {'type': 9, 'data_size': 0.5448831829968969, 'update_interval': 1}, 
    {'type': 1, 'data_size': 0.4236547993389047, 'update_interval': 2}, 
    {'type': 6, 'data_size': 0.6458941130666561, 'update_interval': 2}, 
    {'type': 7, 'data_size': 0.4375872112626925, 'update_interval': 2}, 
    {'type': 3, 'data_size': 0.8917730007820798, 'update_interval': 2}, 
    {'type': 0, 'data_size': 0.9636627605010293, 'update_interval': 2}, 
    {'type': 5, 'data_size': 0.3834415188257777, 'update_interval': 2}]

    views:
    [
    [2, 9], 
    [4, 1], 
    [5], 
    [3], 
    [9, 5], 
    [8, 1], 
    [8, 5], 
    [8, 6], 
    [8, 4], 
    [8]]

    [[7], 
    [0, 4], 
    [0, 5, 9, 2, 3, 4, 8, 7, 1], 
    [5], 
    [5, 4, 7, 9, 1, 3, 8, 6, 0], 
    [1, 2, 7, 3, 6, 5, 9], 
    [7, 4, 1, 3, 8], 
    [1, 7, 8, 4], 
    [5], 
    [3, 8, 6, 4, 7]]

    [7]

    [7]

    [{'type': 0, 'data_size': 0.9636627605010293, 'update_interval': 2}, {'type': 7, 'data_size': 0.4375872112626925, 'update_interval': 2}]
    """