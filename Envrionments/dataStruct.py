import numpy as np

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
        self.slot_length = slot_length
        self._number = int((end - start + 1) / slot_length)
    
    def get_slot_length(self):
        """method to get the length of each time slot"""
        return self.slot_length

    def get_number(self) -> int:
        return self._number

class informationList(object):
    """
    This class is used to store the information list of the environment.
    """
    def __init__(
        self, 
        information_number: int, 
        seed: int, 
        data_size_low_bound: float,
        data_size_up_bound: float,
        data_types_number: int,
        update_interval_low_bound: int,
        update_interval_up_bound: int) -> None:
        """ initialize the information list.
        Args:
            information_number: the number of information list.
            seed: the random seed.
            data_size_low_bound: the low bound of the data size.
            data_size_up_bound: the up bound of the data size.
            data_types_number: the number of data types.
            update_interval_low_bound: the low bound of the update interval.
            update_interval_up_bound: the up bound of the update interval.
        """
        self._number = information_number
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
            self.information_list.append(
                {
                    "type": self.types_of_information[i],
                    "data_size": self.data_size_of_information[i],
                    "update_interval": self.update_interval_of_information[i]
                }
            )
        
    def get_information_list(self) -> list:
        return self.information_list
    
    def set_information_list(self, information_list) -> None:
        self.information_list = information_list

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

    def get_location(self, nowTimeSlot: int) -> location:
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
        max_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        seed: int) -> None:
        """ initialize the vehicle.
        Args:
            vehicle_no: the index of vehicle. e.g. 0, 1, 2, ...
            vehicle_trajectory: the trajectory of the vehicle.
            information_number: the number of information list.
            max_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seed: the random seed.
        """
        self._vehicle_no = vehicle_no
        self._vehicle_trajectory = vehicle_trajectory
        self._information_number = information_number
        self._max_information_number = max_information_number
        self._min_sensing_cost = min_sensing_cost
        self._max_sensing_cost = max_sensing_cost
        self._transmission_power = transmission_power
        self._seed = seed

        if self._max_information_number > self._information_number:
            raise ValueError("The max information number must be less than the information number.")
        
        self.information_canbe_sensed = self.information_types_can_be_sensed()

        self.sensing_cost = self.sensing_cost_of_information()

    def get_vehicle_no(self) -> int:
        return self._vehicle_no

    def get_transmission_power(self) -> float:
        return self._transmission_power

    def information_types_can_be_sensed(self) -> list:
        np.random.seed(self._seed)
        return list(np.random.choice(
            a=self.information_number,
            size=self._max_information_number,
            replace=False))

    def sensing_cost_of_information(self) -> list:
        np.random.seed(self._seed)
        return list(np.random.uniform(
            low=self._min_sensing_cost,
            high=self._max_sensing_cost,
            size=self._max_information_number
        ))
    
    def get_vehicle_location(self, nowTimeSlot: int) -> location:
        return self._vehicle_trajectory.get_location(nowTimeSlot)

    def get_distance_between_edge(self, nowTimeSlot: int, edge_location: location) -> float:
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)

    def set_information_number(self, information_number: int) -> None:
        self._information_number = information_number
    
    def set_max_information_number(self, max_information_number: int) -> None:
        self._max_information_number = max_information_number

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
        vehicles_number: int, 
        vehicle_trajectories: list,
        information_number: int,
        max_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        seeds: list) -> None:
        """ initialize the vehicle list.
        Args:
            vehicles_number: the number of vehicles.
            vehicle_trajectories: the trajectory list.
            information_number: the number of information list.
            max_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seeds: the random seed list.
        """
        self._vehicles_number = vehicles_number
        self._vehicle_trajectories = vehicle_trajectories
        self._information_number = information_number
        self._max_information_number = max_information_number
        self._min_sensing_cost = min_sensing_cost
        self._max_sensing_cost = max_sensing_cost
        self._transmission_power = transmission_power
        self._seeds = seeds

        self._vehicle_list = []
        for i in range(self._vehicles_number):
            self._vehicle_list.append(
                vehicle(
                    vehicle_no=i,
                    vehicle_trajectory=self._vehicle_trajectories[i],
                    information_number=self._information_number,
                    max_information_number=self._max_information_number,
                    min_sensing_cost=self._min_sensing_cost,
                    max_sensing_cost=self._max_sensing_cost,
                    transmission_power=self._transmission_power,
                    seed=self._seeds[i]
                )
            )

    def get_vehicle_list(self) -> list:
        return self._vehicle_list

    def get_vehicles_number(self) -> int:
        return self._vehicles_number

    def get_vehicle(self, vehicle_no: int) -> vehicle:
        return self._vehicle_list[vehicle_no]

    def set_vehicle_list(self, vehicle_list: list) -> None:
        self._vehicle_list = vehicle_list

    def set_vehicles_number(self, vehicles_number: int) -> None:
        self._vehicles_number = vehicles_number


class edge(object):
    """ the edge. """
    def __init__(
        self, 
        edge_no: int,
        edge_location: location,
        communication_range: float,
        bandwidth: float) -> None:
        """ initialize the edge.
        Args:
            edge_no: the index of edge. e.g. 0, 1, 2, ...
            edge_location: the location of the edge.
            communication_range: the range of V2I communications.
            bandwidth: the bandwidth of edge.
        """
        self._edge_no = edge_no
        self._edge_location = edge_location
        self._communication_range = communication_range
        self._bandwidth = bandwidth

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

class applicationList(object):
    """
    This class is used to store the application list of the environment.
    """
    def __init__(
        self, 
        application_number: int,
        view_number: int,
        views_per_application: int,
        seed: int) -> None:
        """ initialize the application list.
        Args:
            application_number: the number of application list.
            view_number: the number of view list.
            seed: the random seed.
        """
        self._number = application_number
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
        view_number: int, 
        information_number: int, 
        max_information_number: int, 
        seeds: list) -> None:
        """ initialize the view list.
        Args:
            view_number: the number of view list.
            information_number: the number of information.
            max_information_number: the maximume number of information required by one view.
            seeds: the random seeds.
        """
        self._number = view_number
        self._information_number = information_number
        self._max_information_number = max_information_number
        self._seeds = seeds

        if self._max_information_number > self._information_number:
            raise ValueError("The max_information_number must be less than the information_number.")

        if len(self._seeds) != self._number:
            raise ValueError("The number of seeds must be equal to the number of view lists.")

        self.view_list = list()

        np.random.seed(self._seeds[0])
        self._random_information_number = np.random.randint(
            size=self._number,
            low=1,
            high=self._max_information_number
        )

        for _ in range(self._number):
            random_information_number = self._random_information_number[_]
            np.random.seed(self._seeds[_])
            self.view_list.append(
                list(np.random.choice(
                    a=self._information_number, 
                    size=random_information_number,
                    replace=False
                ))
            )
        
    def get_view_list(self) -> list:
        """ get the view list.
        Returns:
            the view list.
        """
        return self.view_list

    def set_view_list(self, view_list: list) -> None:
        """ set the view list.
        Args:
            view_list: the view list.
        """
        self.view_list = view_list
