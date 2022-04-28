from Environments.dataStruct import informationList
from Environments.dataStruct import edge, edgeAction
from Environments.dataStruct import vehicle, vehicleAction
import pandas as pd
import numpy as np
import time

class vehicleTrajectoriesProcessor(object):
    def __init__(
        self, 
        file_name: str, 
        longitude_min: float, 
        latitude_min: float, 
        map_width: float,
        time_start: str, 
        time_end: str, 
        out_file: str) -> None:
        """The constructor of the class."""
        """
        Args:
            file_name: the name of the file to be processed. 
                e.g., '/CSV/gps_20161116', source: Didi chuxing gaia open dataset initiative
            longitude_min: the minimum longitude of the bounding box. e.g., 104.04565967220308
            latitude_min: the minimum latitude of the bounding box. e.g., 30.654605745741608
            map_width: the width of the bounding box. e.g., 500 (meters)
            time_start: the start time. e.g., '2016-11-16 08:00:00'
            time_end: the end time. e.g., '2016-11-16 08:05:00'
            out_file: the name of the output file.  e.g., '/CSV/gps_20161116_processed.csv'
        """
        self._file_name = file_name
        self._longitude_min, self._latitude_min = self.gcj02_to_wgs84(longitude_min, latitude_min)
        self.map_width = map_width
        self._time_start = time_start
        self._time_end = time_end
        self._out_file = out_file

        self._longitude_max, self._latitude_max = self.get_longitude_and_latitude_max()

        self.process()

    def get_longitude_and_latitude_max(self) -> tuple:
        longitude_max = self._longitude_min
        latitude_max = self._latitude_min
        precision = 5 * 1e-1   
        """
        += 1e-2 add 1467 meters
        += 1e-3 add 147 meters
        += 1e-4 add 15 meters
        += 1e-5 add 1 meter
        += 1e-6 add 0.25 meters
        """
        length = np.sqrt(2) * self.map_width
        while(True):
            distance = self.get_distance(self._longitude_min, self._latitude_min, longitude_max, latitude_max)
            if np.fabs(distance - length) < precision:
                break
            if np.fabs(distance - length) > 2000.0:
                longitude_max += 1e-2
                latitude_max += 1e-2
            if np.fabs(distance - length) > 150.0 and np.fabs(distance - length) <= 2000.0:
                longitude_max += 1e-3
                latitude_max += 1e-3
            if np.fabs(distance - length) > 15.0 and np.fabs(distance - length) <= 150.0:
                longitude_max += 1e-4
                latitude_max += 1e-4
            if np.fabs(distance - length) > 1.0 and np.fabs(distance - length) <= 15.0:
                longitude_max += 1e-5
                latitude_max += 1e-5
            if np.fabs(distance - length) <= 1.0:
                longitude_max += 1e-6
                latitude_max += 1e-6
        return longitude_max, latitude_max

    def process(self) -> None:

        time_style = "%Y-%m-%d %H:%M:%S"
        time_start_array = time.strptime(self._time_start, time_style)
        time_end_array = time.strptime(self._time_end, time_style)
        time_start = int(time.mktime(time_start_array))
        time_end = int(time.mktime(time_end_array))

        df = pd.read_csv(
            self._file_name, 
            names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'], 
            header=0
        )
        # 经纬度定位
        df.drop(df.columns[[1]], axis=1, inplace=True)
        df.dropna(axis=0)

        df = df[
            (df['longitude'] > self._longitude_min) & 
            (df['longitude'] < self._longitude_max) & 
            (df['latitude'] > self._latitude_min) & 
            (df['latitude'] < self._latitude_max) & 
            (df['time'] > time_start) & 
            (df['time'] < time_end)]  # location
        
        # 排序
        df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)

        vehicle_number = 0
        old_vehicle_id = None
        for index, row in df.iterrows():

            row = dict(df.iloc[index])
            vehicle_id = row['vehicle_id']

            if old_vehicle_id:
                if vehicle_id == old_vehicle_id:
                    row['vehicle_id'] = vehicle_number
                    longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                    row['time'] = row['time'] - time_start
                    x = self.get_distance(self._longitude_min, self._latitude_min, longitude, self._latitude_min)
                    y = self.get_distance(self._longitude_min, self._latitude_min, self._longitude_min, latitude)
                    row['longitude'] = x
                    row['latitude'] = y
                    df.iloc[index] = pd.Series(row)
                else:
                    vehicle_number += 1
                    row['vehicle_id'] = vehicle_number
                    longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                    row['time'] = row['time'] - time_start
                    x = self.get_distance(self._longitude_min, self._latitude_min, longitude, self._latitude_min)
                    y = self.get_distance(self._longitude_min, self._latitude_min, self._longitude_min, latitude)
                    row['longitude'] = x
                    row['latitude'] = y
                    df.iloc[index] = pd.Series(row)
            else:
                row['vehicle_id'] = vehicle_number
                longitude, latitude = self.gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                row['time'] = row['time'] - time_start
                x = self.get_distance(self._longitude_min, self._latitude_min, longitude, self._latitude_min)
                y = self.get_distance(self._longitude_min, self._latitude_min, self._longitude_min, latitude)
                row['longitude'] = x
                row['latitude'] = y
                df.iloc[index] = pd.Series(row)

            old_vehicle_id = vehicle_id

        old_row = None
        for index, row in df.iterrows():
            new_row = dict(df.iloc[index])
            if old_row:
                if old_row['vehicle_id'] == new_row['vehicle_id']:
                    add_number = int(new_row['time']) - int(old_row['time']) - 1
                    if add_number > 0:
                        add_longitude = (float(new_row['longitude']) - float(old_row['longitude'])) / float(add_number)
                        add_latitude = (float(new_row['latitude']) - float(old_row['latitude'])) / float(add_number)
                        for time_index in range(add_number):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [old_row['vehicle_id']],
                                    'time': [old_row['time'] + time_index + 1],
                                    'longitude': [old_row['longitude'] + (time_index + 1) * add_longitude],
                                    'latitude': [old_row['latitude'] + (time_index + 1) * add_latitude]})],
                                axis=0,
                                ignore_index=True)
                else:
                    if old_row['time'] < time_end - time_start:
                        for time_index in range(time_end - time_start - int(old_row['time']) - 1):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [old_row['vehicle_id']],
                                    'time': [old_row['time'] + time_index + 1],
                                    'longitude': [old_row['longitude']],
                                    'latitude': [old_row['latitude']]})],
                                axis=0,
                                ignore_index=True)
                    if new_row['time'] > 0:
                        for time_index in range(int(new_row['time'])):
                            df = pd.concat([df, pd.DataFrame({
                                    'vehicle_id': [new_row['vehicle_id']],
                                    'time': [time_index],
                                    'longitude': [new_row['longitude']],
                                    'latitude': [new_row['latitude']]})],
                                axis=0,
                                ignore_index=True)
                old_row = new_row
            else:
                if new_row['time'] > 0:
                    for time_index in range(int(new_row['time'])):
                        df = pd.concat([df, pd.DataFrame({
                                'vehicle_id': [new_row['vehicle_id']],
                                'time': [time_index],
                                'longitude': [new_row['longitude']],
                                'latitude': [new_row['latitude']]})],
                            axis=0,
                            ignore_index=True)
                old_row = new_row
        df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)
        df.to_csv(self._out_file)

    def get_out_file(self):
        return self._out_file

    def gcj02_to_wgs84(self, lng: float, lat: float):
        """
        GCJ02(火星坐标系)转GPS84
        :param lng:火星坐标系的经度
        :param lat:火星坐标系纬度
        :return:
        """
        a = 6378245.0  # 长半轴
        ee = 0.00669342162296594323

        d_lat = self.trans_form_of_lat(lng - 105.0, lat - 35.0)
        d_lng = self.trans_form_of_lon(lng - 105.0, lat - 35.0)

        rad_lat = lat / 180.0 * np.pi
        magic = np.sin(rad_lat)
        magic = 1 - ee * magic * magic
        sqrt_magic = np.sqrt(magic)

        d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * np.pi)
        d_lng = (d_lng * 180.0) / (a / sqrt_magic * np.cos(rad_lat) * np.pi)
        mg_lat = lat + d_lat
        mg_lng = lng + d_lng
        return [lng * 2 - mg_lng, lat * 2 - mg_lat]

    def trans_form_of_lat(self, lng: float, lat: float):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
            0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
                np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * np.pi) + 320 *
                np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
        return ret

    def trans_form_of_lon(self, lng: float, lat: float):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
            0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * np.pi) + 40.0 *
                np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * np.pi) + 300.0 *
                np.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
        return ret

    def get_distance(self, lng1: float, lat1: float, lng2: float, lat2: float) -> float:
        """ return the distance between two points in meters """
        lng1, lat1, lng2, lat2 = map(np.radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
        d_lon = lng2 - lng1
        d_lat = lat2 - lat1
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
        distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
        distance = round(distance / 1000, 3)
        return distance * 1000

    def get_longitude_min(self) -> float:
        return self._longitude_min
    
    def get_longitude_max(self) -> float:
        return self._longitude_max

    def get_latitude_min(self) -> float:
        return self._latitude_min

    def get_latitude_max(self) -> float:
        return self._latitude_max


class sensingAndQueuing(object):
    """This class is used to get the queue time of the edge with the highest queue length"""
    def __init__(
        self, 
        vehicle: vehicle, 
        vehicle_action: vehicleAction, 
        information_list: informationList
    ) -> None:
        
        self._vehicle_index = vehicle.get_vehicle_index()
        self._sensed_information_number = vehicle.get_sensed_information_number()

        self._action_time = vehicle_action.get_action_time()
        self._sensed_information = vehicle_action.get_sensed_information()
        self._sensing_frequencies = vehicle_action.get_sensing_frequencies()
        self._uploading_priorities = vehicle_action.get_uploading_priorities()

        self._sensed_information_type = vehicle.get_sensed_information_type(self._sensed_information)

        self._arrival_intervals = self.compute_interval_arrival_intervals()
        self._arrival_moments = self.compute_arrival_moments(arrival_intervals=self._arrival_intervals)
        self._updating_moments = self.compute_updating_moments(arrival_moments=self._arrival_moments, information_list=information_list)
        self._queuing_times = self.compute_queuing_times(information_list=information_list)

    def get_sensed_information_type(self) -> np.ndarray:
        return self._sensed_information_type

    def get_arrival_intervals(self) -> np.ndarray:
        return self._arrival_intervals

    def get_arrival_moments(self) -> np.ndarray:
        return self._arrival_moments

    def get_updating_moments(self) -> np.ndarray:
        return self._updating_moments

    def get_queuing_times(self) -> np.ndarray:
        return self._queuing_times

    def compute_interval_arrival_intervals(self) -> np.ndarray:
        """
        Compute the arrival intervals of each information after the action is made
        :return:
            arrival_intervals: a np.ndarray of arrival intervals
        """
        arrival_intervals = np.zeros((self._sensed_information_number,))
        for i in range(self._sensed_information_number):
            if self._sensed_information[i] == 1:
                arrival_intervals[i] = 1 / self._sensing_frequencies[i]
        return arrival_intervals

    def compute_arrival_moments(self, arrival_intervals) -> np.ndarray:
        """
        Compute the arrival moments of each information after the action is made
        the arrival moments are the moments when the information is sensed, 
        i.e., the moments when the information is arrived at the vehicle
        :return:
            arrival_moments: a np.ndarray of arrival moments
        """
        arrival_moments = np.zeros((self._sensed_information_number,))
        for i in range(self._sensed_information_number):
            if self._sensed_information[i] == 1:
                if self._action_time == 0:
                    arrival_moments[i] = 0
                else:
                    arrival_moments[i] = np.floor(self._action_time * self._sensing_frequencies[i]) * arrival_intervals[i]
        return arrival_moments

    def compute_updating_moments(self, arrival_moments: np.ndarray, information_list: informationList) -> np.ndarray:
        """
        Compute the updating moments of each information after the action is made
        the updating moments are the moments when the information is updated, 
        i.e., the moments when the information is updated at the data source
        :return:
            updating_moments: a np.ndarray of updating moments
        """
        updating_moments = np.zeros((self._sensed_information_number,))
        updating_intervals = np.zeros((self._sensed_information_number,))

        for i in range(self._sensed_information_number):
            if self._sensed_information[i] == 1:
                updating_intervals[i] = information_list.get_information_update_interval_by_type(self._sensed_information_type[i])
                if arrival_moments[i] == 0:
                    updating_moments[i] = 0
                else:
                    updating_moments[i] = np.floor(arrival_moments[i] / updating_intervals[i]) * updating_intervals[i]
        return updating_moments
    

    def compute_queuing_times(self, information_list: informationList) -> np.ndarray:
        """
        Compute the queuing times of each information after the action is made
        the queuing times are the moments when the information is queued, 
        i.e., the moments when the information is queued at the vehicle
        :return:
            queuing_times: a np.ndarray of queuing times
        """
        queuing_times = np.zeros((self._sensed_information_number,))

        """sort the actions by the uploading priority"""
        action_list = []
        for i in range(self._sensed_information_number):
            if self._sensed_information[i] == 1:
                action_list.append({
                    "data_type_index": self._sensed_information_type[i],
                    "sensing_frequency": self._sensing_frequencies[i],
                    "uploading_priority": self._uploading_priorities[i]
                })
        action_list.sort(key=lambda value: value["uploading_priority"], reverse=True)
        # print("action_list:\n", action_list)

        # print("mean_service_time:\n", information_list.get_mean_service_time_of_types()[0])
        # print("second_moment_service_time:\n", information_list.get_second_moment_service_time_of_types()[0])

        for index, action in enumerate(action_list):
            data_type_index = int(action["data_type_index"])
            sensing_frequency = action["sensing_frequency"]
            mean_service_time = information_list.get_mean_service_time_by_vehicle_and_type(
                vehicle_index=self._vehicle_index, 
                data_type_index=data_type_index
            )
            second_moment_service_time = information_list.get_second_moment_service_time_by_vehicle_and_type(
                vehicle_index=self._vehicle_index,
                data_type_index=data_type_index
            )
            """if the information is in the head of queue, then the queuing time is 0"""
            if index == 0:
                for i in range(self._sensed_information_number):
                    if self._sensed_information_type[i] == data_type_index:
                        # print("i:", i)
                        # print("mean_service_time:", mean_service_time)
                        # print("second_moment_service_time:", second_moment_service_time)
                        # print("sensing_frequency:", sensing_frequency)
                        queuing_times[i] = mean_service_time + ((sensing_frequency * second_moment_service_time) / (2 * (1 - (sensing_frequency * mean_service_time)))) - mean_service_time
                continue

            """compute the workload and tau before the data type"""
            workload = 0.0
            tau = 0.0
            for i in range(index):
                workload += action_list[i]["sensing_frequency"] * \
                    information_list.get_mean_service_time_by_vehicle_and_type(
                        vehicle_index=self._vehicle_index, 
                        data_type_index=int(action_list[i]["data_type_index"])
                    )
                tau += action_list[i]["sensing_frequency"] * \
                    information_list.get_second_moment_service_time_by_vehicle_and_type(
                        vehicle_index=self._vehicle_index,
                        data_type_index=int(action_list[i]["data_type_index"])
                    )
            workload += sensing_frequency * mean_service_time
            tau += sensing_frequency * second_moment_service_time

            """compute the queuing time"""
            queuing_time = (1.0 / (1.0 - workload + sensing_frequency * mean_service_time)) * \
                (mean_service_time + tau / (2.0 * (1.0 - workload))) - mean_service_time

            for i in range(self._sensed_information_number):
                if self._sensed_information_type[i] == data_type_index:
                    queuing_times[i] = queuing_time

        return queuing_times


class v2iTransmission(object):
    """
    This class is used to define the transmission of a vehicle to an edge.
    """
    def __init__(
        self, 
        vehicle: vehicle, 
        vehicle_action: vehicleAction, 
        edge: edge, 
        edge_action: edgeAction,
        arrival_moments: np.ndarray, 
        queuing_times: np.ndarray,
        white_gaussian_noise: int,
        mean_channel_fading_gain: float,
        second_moment_channel_fading_gain: float,
        path_loss_exponent: int,
        information_list: informationList,
    ) -> None:
        self._vehicle_index = vehicle.get_vehicle_index()
        self._vehicle_trajectory = vehicle.get_vehicle_trajectory()
        self._transmission_power = vehicle_action.get_transmission_power()
        self._edge_location = edge.get_edge_location()
        self._sensed_information_number = vehicle.get_sensed_information_number()
        self._sensed_information = vehicle_action.get_sensed_information()

        self._information_list = information_list
        
        self._sensed_information_type = vehicle.get_sensed_information_type(self._sensed_information)

        self._bandwdith_allocation = edge_action.get_bandwidth_allocation()

        self._arrival_moments = arrival_moments
        self._queuing_times = queuing_times
        self._white_gaussian_noise = white_gaussian_noise
        self._mean_channel_fading_gain = mean_channel_fading_gain
        self._second_moment_channel_fading_gain = second_moment_channel_fading_gain
        self._path_loss_exponent = path_loss_exponent

        self._transmission_times = self.compute_transmission_times()

    def get_transmission_times(self) -> np.ndarray:
        return self._transmission_times

    def compute_transmission_times(self) -> np.ndarray:
        """
        Compute the transmission time of the vehicle to the edge
        :return:
            transmission_time: the transmission time of the vehicle to the edge
        """
        transmission_times = np.zeros((self._sensed_information_number,))
        """compute the transmission time"""
        for i in range(self._sensed_information_number):
            if self._sensed_information[i] == 1:
                start_time = int(np.floor(self._arrival_moments[i] + self._queuing_times[i]))
                vehicle_loaction = self._vehicle_trajectory.get_location(start_time)
                distance = vehicle_loaction.get_distance(self._edge_location)
                SNR = compute_SNR(
                    white_gaussian_noise=self._white_gaussian_noise, 
                    channel_fading_gain=self._mean_channel_fading_gain,
                    # channel_fading_gain=generate_channel_fading_gain(self._mean_channel_fading_gain, self._second_moment_channel_fading_gain),
                    distance=distance,
                    path_loss_exponent=self._path_loss_exponent,
                    transmission_power=self._transmission_power
                )
                tranmission_rate = compute_transmission_rate(
                    SNR, self._bandwdith_allocation[self._vehicle_index])
                transmission_times[i] = self._information_list.get_information_siez_by_type(self._sensed_information_type[i]) / tranmission_rate
        return transmission_times


def get_minimum_transmission_power(
    white_gaussian_noise: int,
    mean_channel_fading_gain: float,
    second_moment_channel_fading_gain: float,
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
    channel_fading_gains = generate_channel_fading_gain(
        mean_channel_fading_gain=mean_channel_fading_gain,
        second_moment_channel_fading_gain=second_moment_channel_fading_gain,
        size=100
    )
    while True:
        probabiliity = compute_successful_tansmission_probability(
            white_gaussian_noise=white_gaussian_noise,
            channel_fading_gains=channel_fading_gains,
            distance=distance,
            path_loss_exponent=path_loss_exponent,
            transmission_power=minimum_transmission_power,
            SNR_target=SNR_target
        )
        if probabiliity <= probabiliity_threshold:
            break
        else:
            minimum_transmission_power -= minimum_transmission_power * 0.01

    return minimum_transmission_power
    

def compute_successful_tansmission_probability(
    white_gaussian_noise: int,
    channel_fading_gains: np.ndarray,
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
    successful_transmission_number = 0
    total_number = 0
    for channel_fading_gain in channel_fading_gains:
        total_number += 1
        SNR = compute_SNR(
            white_gaussian_noise=white_gaussian_noise,
            channel_fading_gain=channel_fading_gain,
            distance=distance,
            path_loss_exponent=path_loss_exponent,
            transmission_power=transmission_power
        )
        if cover_ratio_to_dB(SNR) >= SNR_target:
            successful_transmission_number += 1
    return successful_transmission_number / total_number


def compute_SNR(
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
    return (1.0 / cover_dBm_to_W(white_gaussian_noise)) * \
        np.power(np.abs(channel_fading_gain), 2) * \
        1.0 / (np.power(distance, path_loss_exponent)) * \
        cover_mW_to_W(transmission_power)

def compute_transmission_rate(SNR, bandwidth) -> float:
    """
    :param SNR:
    :param bandwidth:
    :return: transmission rate measure by bit/s
    """
    return float(cover_MHz_to_Hz(bandwidth) * np.log2(1 + SNR))

def generate_channel_fading_gain(mean_channel_fading_gain, second_moment_channel_fading_gain, size: int = 1):
    channel_fading_gain = np.random.normal(loc=mean_channel_fading_gain, scale=second_moment_channel_fading_gain, size=size)
    return channel_fading_gain

def cover_bps_to_Mbps(bps: float) -> float:
    return bps / 1000000

def cover_Mbps_to_bps(Mbps: float) -> float:
    return Mbps * 1000000

def cover_MHz_to_Hz(MHz: float) -> float:
    return MHz * 1e6

def cover_ratio_to_dB(ratio: float) -> float:
    return 10 * np.log10(ratio)

def cover_dB_to_ratio(dB: float) -> float:
    return np.power(10, (dB / 10))

def cover_dBm_to_W(dBm: float) -> float:
    return np.power(10, (dBm / 10)) / 1000

def cover_W_to_dBm(W: float) -> float:
    return 10 * np.log10(W * 1000)

def cover_W_to_mW(W: float) -> float:
    return W * 1000

def cover_mW_to_W(mW: float) -> float:
    return mW / 1000
