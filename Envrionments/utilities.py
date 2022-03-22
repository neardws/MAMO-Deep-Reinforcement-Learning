from dataStruct import vehicle
from dataStruct import edge

import numpy as np


class Queuing():
    # TODO: implement the queueing model
    """This class is used to get the queue time of the edge with the highest queue length"""
    def __init__(self, vehicle_list, edge_list):
        self.vehicle_list = vehicle_list
        self.edge_list = edge_list




class V2I_Transmission():
    # TODO: implement the V2I transmission model
    """
    This class is used to define the transmission of a vehicle to an edge.
    """
    def __init__(self, vehicle, intersection, time_to_arrival):
        self.vehicle = vehicle
        self.intersection = intersection
        self.time_to_arrival = time_to_arrival

    @staticmethod
    def compute_transmission_rate(SNR, bandwidth):
        """
        :param SNR:
        :param bandwidth:
        :return: transmission rate measure by Byte/s
        """
        return int(V2I_Transmission.cover_MHz_to_Hz(bandwidth) * np.log2(1 + SNR) / 8)

    @staticmethod
    def cover_MHz_to_Hz(MHz):
        return MHz * 10e6

    @staticmethod
    def cover_ratio_to_dB(ratio):
        return 10 * np.log10(ratio)

    @staticmethod
    def cover_dB_to_ratio(dB):
        return np.power(10, (dB / 10))

    @staticmethod
    def cover_dBm_to_W(dBm):
        return np.power(10, (dBm / 10)) / 1000

    @staticmethod
    def cover_W_to_dBm(W):
        return 10 * np.log10(W * 1000)

    @staticmethod
    def cover_W_to_mW(W):
        return W * 1000

    @staticmethod
    def cover_mW_to_W(mW):
        return mW / 1000