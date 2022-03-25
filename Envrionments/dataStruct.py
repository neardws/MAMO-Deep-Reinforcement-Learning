import numpy as np
import pandas as pd

from Envrionments.utilities import v2iTransmission
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
        self.slot_length = slot_length
        self._number = int((end - start + 1) / slot_length)
    
    def get_slot_length(self):
        """method to get the length of each time slot"""
        return self.slot_length

    def get_number(self) -> int:
        return self._number


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
        generation_time: float = -1,
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
            generation_time: the generation time of the information.
            inter_arrival_interval: the inter-arrival interval of the information.
            arrival_moment: the arrival moment of the information.
            queuing_time: the queuing time of the information.
            transmission_time: the transmission time of the information.
            received_moment: the received moment of the information.
        """
        self._type = type
        self._vehicle_no = vehicle_no
        self._edge_no = edge_no
        self._generation_time = generation_time
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
    
    def get_generation_time(self) -> float:
        return self._generation_time
    
    def set_generation_time(self, generation_time: float) -> None:
        self._generation_time = generation_time

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

    def get_distance_between_edge(self, nowTimeSlot: int, edge_location) -> float:
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)

    def get_max_information_number(self) -> int:
        return self._max_information_number
    
    def get_information_canbe_sensed(self) -> list:
        return self.information_canbe_sensed
    
    def get_vehicle_trajectory(self) -> trajectory:
        return self._vehicle_trajectory

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
        vehicle_trajectories_file_name: str,
        information_number: int,
        max_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        seeds: list) -> None:
        """ initialize the vehicle list.
        Args:
            vehicles_number: the number of vehicles.
            vehicle_trajectories_file_name: the file name of the vehicle trajectories.
            information_number: the number of information list.
            max_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seeds: the random seed list.
        """
        self._vehicles_number = vehicles_number
        self._vehicle_trajectories_file_name = vehicle_trajectories_file_name
        self._information_number = information_number
        self._max_information_number = max_information_number
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
        vehicle_list: vehicleList,
        sensed_information: list,
        sensing_frequencies: list,
        uploading_priorities: list,
        transmission_power: float, 
        action_time: int) -> None:
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

        if not self.check_action(now_time, vehicle_list):
            raise ValueError("The action is not valid.")

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
            == len(self._uploading_priorities) == len(vehicle.get_max_information_number())):
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
    def generate_from_neural_network_output(network_output: Any):
        """ generate the vehicle action from the neural network output.
        Args:
            network_output: the output of the neural network.
        Returns:
            the vehicle action.
        """
        # TODO: implement this function to generate_from_neural_network_output.
        return vehicleAction(
            vehicle_no=network_output[0],
            sensed_information_types=network_output[1],
            sensing_frequencies=network_output[2],
            uploading_priorities=network_output[3],
            transmission_power=network_output[4],
            action_time=network_output[5]
        )


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


class informationList(object):
    """
    This class is used to store the information list of the environment.
    to store the whole information list, 
    which randomly initialize the characteristics of each information,
    including the type, data size, update interval.
    """
    def __init__(
        self, 
        information_number: int, 
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


    def get_information_list(self) -> list:
        return self.information_list
    
    def set_information_list(self, information_list) -> None:
        self.information_list = information_list
    
    def get_information_by_type(self, type: int) -> dict:
        """method to get the information by type"""
        for information in self.information_list:
            if information["type"] == type:
                return information
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
                infor = self.get_information_by_type(data_type_index)
                for location in vehicle.get_vehicle_trajectory().get_locations():
                    distance = location.get_distance(edge_node)
                    channel_fading_gain = v2iTransmission.generate_channel_fading_gain(
                        mean=mean_channel_fadding_gain,
                        second_moment=second_channel_fadding_gain
                    )
                    SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
                        (1 / np.power(distance, path_loss_exponent)) * \
                        v2iTransmission.cover_mW_to_W(vehicle.get_transmission_power())
                    bandwidth = edge.get_bandwidth() / vehicle_number
                    transmission_time.append(infor["data_size"] / v2iTransmission.compute_transmission_rate(SNR, bandwidth))
                
                mean_service_time = np.array(transmission_time).mean()
                second_moment_service_time = np.array(transmission_time).var()
                mean_service_time_of_types[vehicle_index][data_type_index] = mean_service_time
                second_moment_service_time_of_types[vehicle_index][data_type_index] = second_moment_service_time

        return mean_service_time_of_types, second_moment_service_time_of_types    