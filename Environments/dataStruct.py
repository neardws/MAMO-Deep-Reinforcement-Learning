import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


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
        self._start = start
        self._end = end
        self._slot_length = slot_length
        self._number = int((end - start + 1) / slot_length)
        self._now = start
        self.reset()

    def __str__(self) -> str:
        return f"now time: {self._now}, [{self._start} , {self._end}] with {self._slot_length} = {self._number} slots"
    
    def add_time(self) -> None:
        """method to add time to the system"""
        self._now += 1

    def is_end(self) -> bool:
        """method to check if the system is at the end of the time slots"""
        return self._now >= self._end

    def get_slot_length(self) -> int:
        """method to get the length of each time slot"""
        return int(self._slot_length)

    def get_number(self) -> int:
        return int(self._number)

    def now(self) -> int:
        return int(self._now)
    
    def get_start(self) -> int:
        return int(self._start)

    def get_end(self) -> int:
        return int(self._end)

    def reset(self) -> None:
        self._now = self._start

class informationPacket(object):
    """
    the object of information, which is used to store the information flow,
    including information generation, queuing, transmission, 
    and finally received at the edge.
    """
    def __init__(
        self,
        type: int,
        vehicle_index: int = -1,
        edge_index: int = -1,
        updating_moment: float = -1,
        inter_arrival_interval: float = -1,
        arrival_moment: float = -1,
        queuing_time: float = -1,
        transmission_time: float = -1,
        received_moment: float = -1) -> None:
        """ initialize the information.
        Args:
            type: the type of the information.
            vehicle_index: the index of the vehicle.
            edge_index: the index of the edge.
            updating_moment: the generation time of the information.
            inter_arrival_interval: the inter-arrival interval of the information.
            arrival_moment: the arrival moment of the information.
            queuing_time: the queuing time of the information.
            transmission_time: the transmission time of the information.
            received_moment: the received moment of the information.
        """
        self._type = type
        self._vehicle_index = vehicle_index
        self._edge_index = edge_index
        self._updating_moment = updating_moment
        self._inter_arrival_interval = inter_arrival_interval
        self._arrival_moment = arrival_moment
        self._queuing_time = queuing_time
        self._transmission_time = transmission_time
        self._received_moment = received_moment

    def __str__(self) -> str:
        return f"type: {self._type}\n vehicle_index: {self._vehicle_index}\n edge_index: {self._edge_index}\n updating_moment: {self._updating_moment}\n inter_arrival_interval: {self._inter_arrival_interval}\n arrival_moment: {self._arrival_moment}\n queuing_time: {self._queuing_time}\n transmission_time: {self._transmission_time}\n received_moment: {self._received_moment}"

    def get_type(self) -> int:
        return int(self._type)
    
    def get_vehicle_index(self) -> int:
        return int(self._vehicle_index)
    
    def get_edge_index(self) -> int:
        return int(self._edge_index)
    
    def get_updating_moment(self) -> float:
        return self._updating_moment

    def get_inter_arrival_interval(self) -> float:
        return self._inter_arrival_interval

    def get_arrival_moment(self) -> float:
        return self._arrival_moment
    
    def get_queuing_time(self) -> float:
        return self._queuing_time
    
    def get_transmission_time(self) -> float:
        return self._transmission_time
    
    def get_received_moment(self) -> float:
        return self._received_moment

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

    def __str__(self) -> str:
        return f"x: {self._x}, y: {self._y}"

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
    def __init__(self, timeSlots: timeSlots, locations: List[location]) -> None:
        """ initialize the trajectory.
        Args:
            max_time_slots: the maximum number of time slots.
            locations: the location list.
        """
        self._locations = locations

        if len(self._locations) != timeSlots.get_number():
            raise ValueError("The number of locations must be equal to the max_timestampes.")

    def __str__(self) -> str:
        return str([str(location) for location in self._locations])

    def get_location(self, nowTimeSlot: int) -> location:
        """ get the location of the timestamp.
        Args:
            timestamp: the timestamp.
        Returns:
            the location.
        """
        return self._locations[nowTimeSlot]

    def get_locations(self) -> List[location]:
        """ get the locations.
        Returns:
            the locations.
        """
        return self._locations

    def __str__(self) -> str:
        """ print the trajectory.
        Returns:
            the string of the trajectory.
        """
        print_result= ""
        for index, location in enumerate(self._locations):
            if index % 10 == 0:
                print_result += "\n"
            print_result += "(" + str(index) + ", "
            print_result += str(location.get_x()) + ", "
            print_result += str(location.get_y()) + ")"
        return print_result

class vehicle(object):
    """" the vehicle. """
    def __init__(
        self, 
        vehicle_index: int,
        vehicle_trajectory: trajectory,
        information_number: int,
        sensed_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        seed: int) -> None:
        """ initialize the vehicle.
        Args:
            vehicle_index: the index of vehicle. e.g. 0, 1, 2, ...
            vehicle_trajectory: the trajectory of the vehicle.
            information_number: the number of information list.
            sensed_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seed: the random seed.
        """
        self._vehicle_index = vehicle_index
        self._vehicle_trajectory = vehicle_trajectory
        self._information_number = information_number
        self._sensed_information_number = sensed_information_number
        self._min_sensing_cost = min_sensing_cost
        self._max_sensing_cost = max_sensing_cost
        self._transmission_power = transmission_power
        self._seed = seed

        if self._sensed_information_number > self._information_number:
            raise ValueError("The max information number must be less than the information number.")
        
        self._information_canbe_sensed = self.information_types_can_be_sensed()

        self._sensing_cost = self.sensing_cost_of_information()

    def __str__(self) -> str:
        return f"vehicle_index: {self._vehicle_index}\n vehicle_trajectory: {self._vehicle_trajectory}\n information_number: {self._information_number}\n sensed_information_number: {self._sensed_information_number}\n min_sensing_cost: {self._min_sensing_cost}\n max_sensing_cost: {self._max_sensing_cost}\n transmission_power: {self._transmission_power}\n seed: {self._seed}\n information_canbe_sensed: {self._information_canbe_sensed}\n sensing_cost: {self._sensing_cost}"

    def get_vehicle_index(self) -> int:
        return int(self._vehicle_index)

    def get_transmission_power(self) -> float:
        return self._transmission_power

    def get_sensed_information_type(self, sensed_information: Optional[List[int]]) -> np.ndarray:
        sensed_information_type = np.zeros((self._sensed_information_number,))
        for i in range(self._sensed_information_number):
            if sensed_information[i] == 1:
                sensed_information_type[i] = self.get_information_canbe_sensed()[i]
            else:
                sensed_information_type[i] = -1
        return sensed_information_type

    def information_types_can_be_sensed(self) -> List[int]:
        np.random.seed(self._seed)
        return list(np.random.choice(
            a=self._information_number,
            size=self._sensed_information_number,
            replace=False))

    def sensing_cost_of_information(self) -> List[float]:
        np.random.seed(self._seed)
        return list(np.random.uniform(
            low=self._min_sensing_cost,
            high=self._max_sensing_cost,
            size=self._sensed_information_number
        ))

    def get_sensing_cost(self) -> List[float]:
        return self._sensing_cost
    
    def get_sensing_cost_by_type(self, type: int) -> float:
        for _ in range(self._sensed_information_number):
            if self._information_canbe_sensed[_] == type:
                return self._sensing_cost[_]
        raise ValueError("The type is not in the sensing cost list. type: " + str(type))
    
    def get_vehicle_location(self, nowTimeSlot: int) -> location:
        return self._vehicle_trajectory.get_location(nowTimeSlot)

    def get_distance_between_edge(self, nowTimeSlot: int, edge_location: location) -> float:
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)

    def get_sensed_information_number(self) -> int:
        return self._sensed_information_number
    
    def get_information_canbe_sensed(self) -> List[int]:
        return self._information_canbe_sensed

    def get_information_type_canbe_sensed(self, index: int) -> int:
        return self._information_canbe_sensed[index]
    
    def get_vehicle_trajectory(self) -> trajectory:
        return self._vehicle_trajectory

class vehicleList(object):
    """ the vehicle list. """
    def __init__(
        self, 
        number: int, 
        time_slots: timeSlots,
        trajectories_file_name: str,
        information_number: int,
        sensed_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        seeds: List[int]) -> None:
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

        self._vehicle_trajectories = self.read_vehicle_trajectories(time_slots)

        self._vehicle_list: List[vehicle] = []
        for i in range(self._number):
            self._vehicle_list.append(
                vehicle(
                    vehicle_index=i,
                    vehicle_trajectory=self._vehicle_trajectories[i],
                    information_number=self._information_number,
                    sensed_information_number=self._sensed_information_number,
                    min_sensing_cost=self._min_sensing_cost,
                    max_sensing_cost=self._max_sensing_cost,
                    transmission_power=self._transmission_power,
                    seed=self._seeds[i]
                )
            )

    def __str__(self) -> str:
        return f"number: {self._number}\n information_number: {self._information_number}\n sensed_information_number: {self._sensed_information_number}\n min_sensing_cost: {self._min_sensing_cost}\n max_sensing_cost: {self._max_sensing_cost}\n transmission_power: {self._transmission_power}\n seeds: {self._seeds}\n vehicle_list: {self._vehicle_list}" + "\n" + str([str(vehicle) for vehicle in self._vehicle_list])

    def get_vehicle_list(self) -> List[vehicle]:
        return self._vehicle_list

    def get_number(self) -> int:
        return int(self._number)

    def get_sensed_information_number(self) -> int:
        return int(self._sensed_information_number)

    def get_vehicle(self, vehicle_index: int) -> vehicle:
        return self._vehicle_list[vehicle_index]

    def get_vehicle_trajectories(self) -> List[trajectory]:
        return self._vehicle_trajectories

    def read_vehicle_trajectories(self, timeSlots: timeSlots) -> List[trajectory]:

        df = pd.read_csv(
            self._trajectories_file_name, 
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
        vehicle_trajectories: List[trajectory] = []
        for vehicle_id in selected_vehicle_id[ : self._number]:
            new_df = df[df['vehicle_id'] == vehicle_id["vehicle_id"]]
            loc_list: List[location] = []
            for row in new_df.itertuples():
                # time = getattr(row, 'time')
                x = getattr(row, 'longitude')
                y = getattr(row, 'latitude')
                loc = location(x, y)
                loc_list.append(loc)
            new_vehicle_trajectory: trajectory = trajectory(
                timeSlots=timeSlots,
                locations=loc_list
            )
            new_vehicle_id += 1
            vehicle_trajectories.append(new_vehicle_trajectory)

        return vehicle_trajectories

class edge(object):
    """ the edge. """
    def __init__(
        self, 
        edge_index: int,
        information_number: int,
        edge_location: location,
        communication_range: float,
        bandwidth: float) -> None:
        """ initialize the edge.
        Args:
            edge_index: the index of edge. e.g. 0, 1, 2, ...
            information_number: the number of information list.
            edge_location: the location of the edge.
            communication_range: the range of V2I communications.
            bandwidth: the bandwidth of edge.
        """
        self._edge_index = edge_index
        self._information_number = information_number
        self._edge_location = edge_location
        self._communication_range = communication_range
        self._bandwidth = bandwidth

    def get_edge_index(self) -> int:
        return int(self._edge_index)

    def get_edge_location(self) -> location:
        return self._edge_location

    def get_communication_range(self) -> float:
        return self._communication_range
    
    def get_bandwidth(self) -> float:
        return self._bandwidth

class edgeAction(object):
    """ the action of the edge. """
    def __init__(
        self, 
        edge: edge,
        now_time: int,
        vehicle_number: int,
        bandwidth_allocation: np.ndarray,
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

    def __str__(self) -> str:
        return f"edge_bandwidth: {self._edge_bandwidth}\n now_time: {self._now_time}\n vehicle_number: {self._vehicle_number}\n action_time: {self._action_time}\n bandwidth_allocation: {self._bandwidth_allocation}"

    def get_bandwidth_allocation(self) -> np.ndarray:
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
            print("the action time is not correct.")
            return False
        if self._vehicle_number != len(self._bandwidth_allocation):
            print("the number of vehicles is not correct.")
            return False
        if self._edge_bandwidth < self.get_the_sum_of_bandwidth_allocation():
            print("the allocated bandwidth exceeds its cability.")
            print("the allocated bandwidth:", self.get_the_sum_of_bandwidth_allocation())
            print("the edge bandwidth:", self._edge_bandwidth)
            return False
        return True

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
        self._application_list: List[int] = []
        
        if self._views_per_application == 1:
            if self._number != self._view_number:
                self._number = self._view_number
            np.random.seed(self._seed)
            self._application_list = list(np.random.permutation(list(range(self._number))))
        elif self._views_per_application > 1:
            # TODO: to generate the mapping between application and view list.
            pass
        else:
            raise Exception("The views_per_application must be greater than 1.")

    def __str__(self) -> str:
        return f"number: {self._number}\n view_number: {self._view_number}\n views_per_application: {self._views_per_application}\n seed: {self._seed}\n application_list: {self._application_list}"

    def get_number(self) -> int:
        return int(self._number)

    def get_application_list(self) -> List[int]:
        if self._application_list is None:
            raise Exception("The application list is not list.")
        return self._application_list

    def get_view_by_application_index(self, index: int) -> int:
        if self._application_list is None:
            raise Exception("The application list is not list.")
        if index < 0 or index >= self._number:
            raise Exception("The index is out of range.")
        return self._application_list[index]

class viewList(object):
    """ the view list. """
    def __init__(
        self, 
        number: int, 
        information_number: int, 
        required_information_number: int, 
        seeds: List[int]) -> None:
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

        self._view_list: List[List[int]] = []

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

    def __str__(self) -> str:
        return f"number: {self._number}\n information_number: {self._information_number}\n required_information_number: {self._required_information_number}\n seeds: {self._seeds}\n view_list: {self._view_list}"

    def get_number(self) -> int:
        return int(self._number)
        
    def get_view_list(self) -> List[List[int]]:
        """ get the view list.
        Returns:
            the view list.
        """
        return self._view_list

    def get_information_required_by_view_index(self, index: int) -> List[int]:
        """ get the information required by the view.
        Args:
            index: the index of the view.
        Returns:
            the information required by the view.
        """
        if index < 0 or index >= self._number:
            raise Exception("The index is out of range.")
        return self._view_list[index]

class information(object):
    """
    This class is used to store the information of the environment.
    "type": self.types_of_information[i],
                "data_size": self.data_size_of_information[i],
                "update_interval": self.update_interval_of_information[i]
    """
    def __init__(
        self, 
        type: int, 
        data_size: float,
        update_interval: float) -> None:
        """ initialize the information.
        Args:
            type: the type of the information.
            data_size: the data size of the information.
            update_interval: the update interval of the information.
        """
        self._type = type
        self._data_size = data_size
        self._update_interval = update_interval

    def __str__(self) -> str:
        return f"type: {self._type}\n data_size: {self._data_size}\n update_interval: {self._update_interval}"

    def get_type(self) -> int:
        return int(self._type)
    
    def get_data_size(self) -> float:
        return self._data_size
    
    def get_update_interval(self) -> float:
        return self._update_interval

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
        update_interval_low_bound: float,
        update_interval_up_bound: float,
        vehicle_list: vehicleList,
        edge_node: edge,
        white_gaussian_noise: int,
        mean_channel_fading_gain: float,
        second_moment_channel_fading_gain: float,
        path_loss_exponent: int) -> None:
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
        self._types_of_information: List[int] = np.random.permutation(
            list(range(self._data_types_number))
        )

        np.random.seed(self._seed)
        self._data_size_of_information: List[float] = np.random.uniform(
            low=self._data_size_low_bound,
            high=self._data_size_up_bound,
            size=self._number,
        )

        np.random.seed(self._seed)
        self._update_interval_of_information: List[float] = np.random.uniform(
            low=self._update_interval_low_bound, 
            high=self._update_interval_up_bound,
            size=self._number, 
        )

        self._information_list: List[information] = []
        for i in range(self._number):
            self._information_list.append(
                information(
                    type=self._types_of_information[i],
                    data_size=self._data_size_of_information[i],
                    update_interval=self._update_interval_of_information[i]
                )
            )
        
        self._mean_service_time_of_types, self._second_moment_service_time_of_types = \
            self.compute_mean_and_second_moment_service_time_of_types(
                vehicle_list=vehicle_list,
                edge_node=edge_node,
                white_gaussian_noise=white_gaussian_noise,
                mean_channel_fading_gain=mean_channel_fading_gain,
                second_moment_channel_fading_gain=second_moment_channel_fading_gain,
                path_loss_exponent=path_loss_exponent
            )

    def __str__(self) -> str:
        return f"number: {self._number}\n seed: {self._seed}\n data_size_low_bound: {self._data_size_low_bound}\n data_size_up_bound: {self._data_size_up_bound}\n data_types_number: {self._data_types_number}\n update_interval_low_bound: {self._update_interval_low_bound}\n update_interval_up_bound: {self._update_interval_up_bound}\n types_of_information: {self._types_of_information}\n data_size_of_information: {self._data_size_of_information}\n update_inter_of_information: {self._update_interval_of_information}"

    def get_number(self) -> int:
        """ get the number of information.
        Returns:
            the number of information.
        """
        return int(self._number)

    def get_information_list(self) -> List[information]:
        return self._information_list
    
    def get_information_type_by_index(self, index: int) -> int:
        if index >= self._number:
            raise ValueError("The index is out of range.")
        return self._information_list[index].get_type()

    def get_information_by_type(self, type: int) -> information:
        """method to get the information by type"""
        for infor in self._information_list:
            if infor.get_type() == type:
                return infor
        raise ValueError("The type is not in the list.")

    def get_information_siez_by_type(self, type: int) -> float:
        """method to get the information size by type"""
        for information in self._information_list:
            if information.get_type() == type:
                return information.get_data_size()
        raise ValueError("The type is not in the list.")

    def get_information_update_interval_by_type(self, type: int) -> float:
        """method to get the information update interval by type"""
        for information in self._information_list:
            if information.get_type() == type:
                return information.get_update_interval()
        raise ValueError("The type is not in the list.")

    def get_mean_service_time_of_types(self) -> np.ndarray:
        return self._mean_service_time_of_types
    
    def get_second_moment_service_time_of_types(self) -> np.ndarray:
        return self._second_moment_service_time_of_types

    def get_mean_service_time_by_vehicle_and_type(self, vehicle_index: int, data_type_index: int) -> float:
        return self._mean_service_time_of_types[vehicle_index][data_type_index]

    def get_second_moment_service_time_by_vehicle_and_type(self, vehicle_index: int, data_type_index: int) -> float:
        return self._second_moment_service_time_of_types[vehicle_index][data_type_index]

    def compute_mean_and_second_moment_service_time_of_types(
        self, 
        vehicle_list: vehicleList,
        edge_node: edge,
        white_gaussian_noise,
        mean_channel_fading_gain,
        second_moment_channel_fading_gain,
        path_loss_exponent) -> Tuple[np.ndarray, np.ndarray]:
        """
        method to get the mean and second moment service time of 
        each type of information at each vehile.
        Args:
            vehicle_list: the vehicle list.
            edge_node: the edge node.
            white_gaussian_noise: the additive white gaussian noise.
            mean_channel_fading_gain: the mean channel fadding gain.
            second_moment_channel_fading_gain: the second channel fadding gain.
            path_loss_exponent: the path loss exponent.
        Returns:
            the mean and second moment service time of each type of information.
        """
        from Environments.utilities import generate_channel_fading_gain, compute_SNR, compute_transmission_rate

        vehicle_number = vehicle_list.get_number()
        mean_service_time_of_types = np.zeros(shape=(vehicle_number, self._data_types_number))
        second_moment_service_time_of_types = np.zeros(shape=(vehicle_number, self._data_types_number))

        for vehicle_index in range(vehicle_number):
            vehicle = vehicle_list.get_vehicle(vehicle_index)
            for data_type_index in range(self._data_types_number):
                transmission_time = []
                for location in vehicle.get_vehicle_trajectory().get_locations():
                    distance = location.get_distance(edge_node.get_edge_location())
                    channel_fading_gain = generate_channel_fading_gain(
                        mean_channel_fading_gain=mean_channel_fading_gain,
                        second_moment_channel_fading_gain=second_moment_channel_fading_gain
                    )
                    SNR = compute_SNR(
                        white_gaussian_noise=white_gaussian_noise,
                        channel_fading_gain=channel_fading_gain,
                        distance=distance,
                        path_loss_exponent=path_loss_exponent,
                        transmission_power=vehicle.get_transmission_power()
                    )
                    bandwidth = edge_node.get_bandwidth() / vehicle_number
                    if self.get_information_siez_by_type(data_type_index) / compute_transmission_rate(SNR, bandwidth) != np.Inf:
                        transmission_time.append(self.get_information_siez_by_type(data_type_index) / compute_transmission_rate(SNR, bandwidth))
                mean_service_time = np.array(transmission_time).mean()
                second_moment_service_time = np.array(transmission_time).var()
                mean_service_time_of_types[vehicle_index][data_type_index] = mean_service_time
                second_moment_service_time_of_types[vehicle_index][data_type_index] = second_moment_service_time

        return mean_service_time_of_types, second_moment_service_time_of_types

class vehicleAction(object):
    """ the action of the vehicle. """
    def __init__(
        self, 
        vehicle_index: int,
        now_time: int,
        sensed_information: Optional[List[int]] = None,
        sensing_frequencies: Optional[List[float]] = None,
        uploading_priorities: Optional[List[float]] = None,
        transmission_power: Optional[float] = None, 
        action_time: Optional[int] = None) -> None:
        """ initialize the vehicle action.
        Args:
            vehicle_index: the index of vehicle. e.g. 0, 1, 2, ...
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
        self._vehicle_index = vehicle_index
        self._now_time = now_time
        self._sensed_information = sensed_information
        self._sensing_frequencies = sensing_frequencies
        self._uploading_priorities = uploading_priorities
        self._transmission_power = transmission_power
        self._action_time = action_time
    
    def __str__(self) -> str:
        return f"vehicle_index: {self._vehicle_index}, now_time: {self._now_time}, sensed_information: {self._sensed_information}, sensing_frequencies: {self._sensing_frequencies}, uploading_priorities: {self._uploading_priorities}, transmission_power: {self._transmission_power}, action_time: {self._action_time}"

    def check_action(self, nowTimeSlot: int, vehicle_list: vehicleList) -> bool:
        """ check the action.
        Args:
            nowTimeSlot: the time of the action.
        Returns:
            True if the action is valid.
        """
        if self._action_time != nowTimeSlot:
            return False
        if self._vehicle_index >= len(vehicle_list.get_vehicle_list()):
            return False
        vehicle = vehicle_list.get_vehicle(self._vehicle_index)
        if not (len(self._sensed_information) == len(self._sensing_frequencies) == len(self._uploading_priorities)):
            return False
        if self._transmission_power > vehicle.get_transmission_power():
            return False
        return True

    def get_sensed_information(self) -> List[int]:
        return self._sensed_information

    def get_sensing_frequencies(self) -> List[float]:
        return self._sensing_frequencies
    
    def get_uploading_priorities(self) -> List[float]:
        return self._uploading_priorities

    def get_transmission_power(self) -> float:
        return self._transmission_power

    def get_action_time(self) -> int:
        return self._action_time


class informationRequirements(object):
    """
    This class is used to store the data requirements of the environment.
    """
    def __init__(
        self,
        time_slots: timeSlots,
        max_application_number: int,
        min_application_number: int,
        application_list: applicationList,
        view_list: viewList,
        information_list: informationList,
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
        self._application_number = application_list.get_number()
        self._application_list = application_list
        self._view_list = view_list
        self._information_list = information_list
        self._seed = seed

        self.applications_at_time = self.applications_at_times()

    def __str__(self) -> str:
        """ return the string representation of the information set.
        Returns:
            the string representation of the information set.
        """
        return "information requirements: \n" + \
            "max_timestampes: " + str(self._max_timestampes) + "\n" + \
            "max_application_number: " + str(self._max_application_number) + "\n" + \
            "min_application_number: " + str(self._min_application_number) + "\n" + \
            "application_number: " + str(self._application_number) + "\n" + \
            "application_list: " + str(self._application_list) + "\n" + \
            "view_list: " + str(self._view_list) + "\n" + \
            "information_list: " + str(self._information_list) + "\n" + "seed: " + str(self._seed) + "\n"
        
    def get_seed(self) -> int:
        """ get the random seed.
        Returns:
            the random seed.
        """
        return int(self._seed)
    
    def get_applications_at_times(self) -> List[List[int]]:
        """ get the applications at each time.
        Returns:
            the applications at each time.
        """
        return self.applications_at_time

    def applications_at_times(self) -> List[List[int]]:
        """ get the applications at each time.
        Returns:
            the applications at times.
        """
        random_application_number = np.random.randint(
            low=self._min_application_number, 
            high=self._max_application_number, 
            size=self._max_timestampes
        )

        applications_at_times: List[List[int]] = []
        for _ in range(self._max_timestampes):
            applications_at_times.append(
                list(np.random.choice(
                    list(range(self._application_number)), 
                    random_application_number[_], 
                    replace=False))
            )

        return applications_at_times
    
    def applications_at_now(self, nowTimeStamp: int) -> List[int]:
        """ get the applications now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the applications list.
        """
        if self.applications_at_time is None:
            return Exception("applications_at_time is None.")
        return self.applications_at_time[nowTimeStamp]

    def views_required_by_application_at_now(self, nowTimeStamp: int) -> List[int]:
        """ get the views required by application now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the views required by application list.
        """
        applications_at_now = self.applications_at_now(nowTimeStamp)
        views_required_by_application_at_now: List[int] = []
        for _ in applications_at_now:
            views_required_by_application_at_now.append(
                self._application_list.get_view_by_application_index(_)
            )
        return views_required_by_application_at_now

    def get_views_required_number_at_now(self, nowTimeStamp: int) -> int:
        """ get the views required number at now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the views required number.
        """
        views_required_by_application_at_now = self.views_required_by_application_at_now(nowTimeStamp)
        return len(views_required_by_application_at_now)
    
    def get_information_type_required_by_views_at_now_at_now(self, nowTimeStamp: int) -> List[List[int]]:
        """ get the information required by views now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the information set required by views list.
        """
        views_required_by_application_at_now = self.views_required_by_application_at_now(nowTimeStamp)
        
        information_type_required_by_views_at_now: List[List[int]] = []

        for _ in views_required_by_application_at_now:
            information_type_required: List[int] = []
            information_required = self._view_list.get_information_required_by_view_index(_)
            for information_index in information_required:
                information_type_required.append(
                    self._information_list.get_information_type_by_index(information_index)
                ) 
            information_type_required_by_views_at_now.append(
                information_type_required
            )

        return information_type_required_by_views_at_now
    
    def get_information_required_at_now(self, nowTimeStamp: int) -> np.ndarray:
        """ get the information required now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the information set required.
        """
        views_required_by_application_at_now = self.views_required_by_application_at_now(nowTimeStamp)
        
        information_type_required_at_now = np.zeros(self._information_list.get_number())

        for view_index in views_required_by_application_at_now:
            information_required = self._view_list.get_information_required_by_view_index(view_index)
            for information_index in information_required:
                information_type_required_at_now[self._information_list.get_information_type_by_index(information_index)] = 1 

        return information_type_required_at_now
