import numpy as np

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


class trajectories(object):
    """ the trajectories of the node. """
    def __init__(self, max_timestampes: int, locations: list) -> None:
        """ initialize the trajectories.
        Args:
            max_timestampes: the maximum number of timestampes.
            locations: the location list.
        """
        self._max_timestampes = max_timestampes
        self._locations = locations

        if len(self._locations) != self._max_timestampes:
            raise ValueError("The number of locations must be equal to the max_timestampes.")

    def get_location(self, timestamp: int) -> location:
        """ get the location of the timestamp.
        Args:
            timestamp: the timestamp.
        Returns:
            the location.
        """
        return self._locations[timestamp]

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
        vehicle_trajectories: trajectories,
        information_number: int,
        max_information_number: int,
        max_sensing_cost: float,
        transmission_power: float,
        seed: int) -> None:
        """ initialize the vehicle.
        Args:
            vehicle_no: the index of vehicle. e.g. 0, 1, 2, ...
            vehicle_trajectories: the trajectory of the vehicle.
            information_number: the number of information list.
            max_information_number: the maximum number of information, which can be sensed by the vehicle.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seed: the random seed.
        """
        self._vehicle_no = vehicle_no
        self._vehicle_trajectories = vehicle_trajectories
        self._information_number = information_number
        self._max_information_number = max_information_number
        self._max_sensing_cost = max_sensing_cost
        self._transmission_power = transmission_power
        self._seed = seed

        if self._max_information_number > self._information_number:
            raise ValueError("The max information number must be less than the information number.")
        
        self.information_canbe_sensed = self.information_types_can_be_sensed()

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
    
    def get_vehicle_location(self, timestamp: int) -> location:
        return self._vehicle_trajectories.get_location(timestamp)

    def get_distance_between_edge(self, timestamp: int, edge_location: location) -> float:
        return self._vehicle_trajectories.get_location(timestamp).get_distance(edge_location)
        

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

    