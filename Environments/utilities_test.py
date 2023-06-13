import pytest
import numpy as np
from Test.environmentConfig_test import vehicularNetworkEnvConfig
from Environments.utilities import vehicleTrajectoriesProcessor, v2iTransmission, sensingAndQueuing
from Environments.utilities import get_minimum_transmission_power, generate_channel_fading_gain
from Environments.utilities import compute_SNR, compute_transmission_rate, compute_successful_tansmission_probability
from Environments.utilities import cover_MHz_to_Hz, cover_ratio_to_dB, cover_dB_to_ratio, cover_dBm_to_W, cover_W_to_dBm, cover_W_to_mW, cover_mW_to_W, cover_bps_to_Mbps
from Log.logger import myapp
from Environments.dataStruct_test import vehicle_list, information_list, v_action, e_action, edge_node

config = vehicularNetworkEnvConfig()

@pytest.mark.skip(reason="Passed.")
def test_vehicleTrajectoriesProcessor():
    trajectory_processor = vehicleTrajectoriesProcessor(
        file_name='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/gps_20161116', 
        longitude_min=104.04565967220308, 
        latitude_min=30.654605745741608, 
        map_width=1000.0,
        time_start='2016-11-16 08:00:00', 
        time_end='2016-11-16 08:05:00', 
        out_file='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/trajectories_20161116_0800_0850.csv',
    )
    print("longitude_min:\n", trajectory_processor.get_longitude_min())
    print("longitude_max:\n", trajectory_processor.get_longitude_max())
    print("latitude_min:\n", trajectory_processor.get_latitude_min())
    print("latitude_max:\n", trajectory_processor.get_latitude_max())
    
    # new_longitude_min, new_latitude_min = trajectory_processor.gcj02_to_wgs84(trajectory_processor._longitude_min, trajectory_processor._latitude_min)
    print("distance:\n", trajectory_processor.get_distance(
        lng1=trajectory_processor.get_longitude_min(),
        lat1=trajectory_processor.get_latitude_min(),
        lng2=trajectory_processor.get_longitude_max(),
        lat2=trajectory_processor.get_latitude_max()
    ) / np.sqrt(2))

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("MHz", [1, ]) 
def test_cover_MHz_to_Hz(MHz):
    assert cover_MHz_to_Hz(MHz) == 1000000

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("ratio", [10, ]) 
def test_cover_ratio_to_dB(ratio):
    assert cover_ratio_to_dB(ratio) == 10

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("dB", [10, ]) 
def test_cover_dB_to_ratio(dB):
    assert cover_dB_to_ratio(dB) == 10

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("dBm", [10, ]) 
def test_cover_dBm_to_W(dBm):
    assert cover_dBm_to_W(dBm) == 0.01

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("W", [10, ]) 
def test_cover_W_to_dBm(W):
    assert cover_W_to_dBm(W) == 40

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("W", [1, ]) 
def test_cover_W_to_mW(W):
    assert cover_W_to_mW(W) == 1000

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("mW", [1, ]) 
def test_cover_mW_to_W(mW):
    assert cover_mW_to_W(mW) == 0.001

@pytest.mark.skip(reason="Passed.")
def test_compute_SNR():
    for distance in range(10, 510, 10):
        snr = compute_SNR(
            white_gaussian_noise=-90,
            channel_fading_gain=config.mean_channel_fading_gain,
            distance=distance,
            path_loss_exponent=config.path_loss_exponent,
            transmission_power=100
        )
        print("Distance: {}, SNR: {}".format(distance, cover_ratio_to_dB(snr)))

@pytest.mark.skip(reason="Passed.")
def test_generate_channel_fading_gain():
    print(generate_channel_fading_gain(
        mean_channel_fading_gain=config.mean_channel_fading_gain,
        second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
        size=1
    ))
    print(generate_channel_fading_gain(
        mean_channel_fading_gain=config.mean_channel_fading_gain,
        second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
        size=100
    ))

@pytest.mark.skip(reason="Passed.")
def test_compute_transmission_rate():
    snr = compute_SNR(
        white_gaussian_noise=-90,
        channel_fading_gain=config.mean_channel_fading_gain,
        distance=500.0,
        path_loss_exponent=config.path_loss_exponent,
        transmission_power=100
    )
    assert cover_ratio_to_dB(snr) == pytest.approx(35.051499783199056) # dB, SNR = 3200
    assert cover_bps_to_Mbps(compute_transmission_rate(SNR=snr, bandwidth=1)) == pytest.approx(11.64430696) # Mbps, bandwidth = 1 MHz

@pytest.mark.skip(reason="Passed.")
def test_get_minimum_transmission_power():
    power = get_minimum_transmission_power(
                white_gaussian_noise=config.white_gaussian_noise,
                mean_channel_fading_gain=config.mean_channel_fading_gain,
                second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
                distance=500,
                path_loss_exponent=config.path_loss_exponent,
                transmission_power=200,
                SNR_target=35.0,
                probabiliity_threshold=config.probabiliity_threshold)
    print(power)

@pytest.mark.skip(reason="Passed.")
def test_compute_successful_tansmission_probability():
    channel_fading_gains = generate_channel_fading_gain(
        mean_channel_fading_gain=config.mean_channel_fading_gain,
        second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
        size=100
    )
    pro = compute_successful_tansmission_probability(
            white_gaussian_noise=config.white_gaussian_noise,
            channel_fading_gains=channel_fading_gains,
            distance=100,
            path_loss_exponent=config.path_loss_exponent,
            transmission_power=1,
            SNR_target=30.0)
    print(pro)

saq = sensingAndQueuing(
    vehicle=vehicle_list.get_vehicle(vehicle_index=0), 
    vehicle_action=v_action,
    information_list=information_list,
)

@pytest.mark.skip(reason="Passed.")
def test_sensingAndQueuing():
    
    print("\n")
    print(v_action.get_sensed_information())
    print(v_action.get_sensing_frequencies())
    print(v_action.get_uploading_priorities())
    print(v_action.get_action_time())

    sensed_information_type = vehicle_list.get_vehicle(vehicle_index=0).get_sensed_information_type(v_action.get_sensed_information())

    print("\n")
    print("Sensed information type: {}".format(sensed_information_type))

    assert list(saq.get_arrival_intervals()) == [ 1 / frequency for frequency in v_action.get_sensing_frequencies() ]
    for i in range(len(saq.get_arrival_intervals())):
        assert saq.get_arrival_moments()[i] == pytest.approx(np.floor(10 * v_action.get_sensing_frequencies()[i]) * saq.get_arrival_intervals()[i])
        assert saq.get_updating_moments()[i] == pytest.approx(np.floor(saq.get_arrival_moments()[i] / information_list.get_information_update_interval_by_type(sensed_information_type[i])) * information_list.get_information_update_interval_by_type(sensed_information_type[i]))

    # The first updated packet
    print("\n First updated packet:")
    print("mean_service_time:", information_list.get_mean_service_time_of_types()[0][4])
    print("second_moment_service_time:", information_list.get_second_moment_service_time_of_types()[0][4])
    print("sensing_frequency:", v_action.get_sensing_frequencies()[2])
    
    print("\n")
    print("saq.get_queuing_times()[2]:", (1.3426327547947166 + ((0.2753798709891763 * 0.10087165260618523) / (2.0 * (1.0 - (0.2753798709891763 * 1.3426327547947166))))) - 1.3426327547947166 )
    assert saq.get_queuing_times()[2] == information_list.get_mean_service_time_of_types()[0][4] + (v_action.get_sensing_frequencies()[2] * information_list.get_second_moment_service_time_of_types()[0][4]) / (2.0 * (1.0 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4])) - information_list.get_mean_service_time_of_types()[0][4]
    print("saq.get_queuing_times()[1]:", (((1 / (1 - (0.26793181792754145 * 1.3426327547947166))) * (1.5852099694213786 + (((0.26793181792754145 * 0.10087165260618523) + (0.1804523162539861 * 0.13492532877001698)) / (2 * ((1 - (0.26793181792754145 * 1.3426327547947166)) - (0.1804523162539861 * 1.5852099694213786)))))) - 1.5852099694213786))
    assert pytest.approx(saq.get_queuing_times()[1]) == pytest.approx(((1 / (1 - (0.26793181792754145 * 1.3426327547947166))) * (1.5852099694213786 + (((0.26793181792754145 * 0.10087165260618523) + (0.1804523162539861 * 0.13492532877001698)) / (2 * ((1 - (0.26793181792754145 * 1.3426327547947166)) - (0.1804523162539861 * 1.5852099694213786)))))) - 1.5852099694213786)
    assert pytest.approx(saq.get_queuing_times()[1]) == pytest.approx((1 / (1 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4])) * \
        (information_list.get_mean_service_time_of_types()[0][8] + ((v_action.get_sensing_frequencies()[2] * information_list.get_second_moment_service_time_of_types()[0][4] + v_action.get_sensing_frequencies()[1] * information_list.get_second_moment_service_time_of_types()[0][8]) / (2 * (1 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4] - v_action.get_sensing_frequencies()[1] * information_list.get_mean_service_time_of_types()[0][8])))) - information_list.get_mean_service_time_of_types()[0][8])
    # print("saq.get_queuing_times()[0]:", ((1 / ((1 - (0.26793181792754145 * 1.3426327547947166)) - (0.1804523162539861 * 1.5852099694213786))) * (1.2234147523336023 + ((((0.26793181792754145 * 0.10087165260618523) + (0.1804523162539861 * 0.13492532877001698)) + (0.1804523162539861 * 0.08334924184781246)) / (2 * (((1 - (0.26793181792754145 * 1.3426327547947166)) - (0.1804523162539861 * 1.5852099694213786)) - (0.265005104646134 * 1.2234147523336023)))))) - information_list.get_mean_service_time_of_types()[0][2])
    # print("part1:", (1 / (1 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4] - v_action.get_sensing_frequencies()[1] * information_list.get_mean_service_time_of_types()[0][8])))
    # print("part2:", (information_list.get_mean_service_time_of_types()[0][2] + ((v_action.get_sensing_frequencies()[2] * information_list.get_second_moment_service_time_of_types()[0][4] + v_action.get_sensing_frequencies()[1] * information_list.get_second_moment_service_time_of_types()[0][8] + v_action.get_sensing_frequencies()[1] * information_list.get_second_moment_service_time_of_types()[0][2]) / (2 * (1 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4] - v_action.get_sensing_frequencies()[1] * information_list.get_mean_service_time_of_types()[0][8] - v_action.get_sensing_frequencies()[0] * information_list.get_mean_service_time_of_types()[0][2])))))
    # print("part3:", (v_action.get_sensing_frequencies()[2] * information_list.get_second_moment_service_time_of_types()[0][4] + v_action.get_sensing_frequencies()[1] * information_list.get_second_moment_service_time_of_types()[0][8] + v_action.get_sensing_frequencies()[1] * information_list.get_second_moment_service_time_of_types()[0][2]))
    # print("part4:", (1 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4] - v_action.get_sensing_frequencies()[1] * information_list.get_mean_service_time_of_types()[0][8] - v_action.get_sensing_frequencies()[0] * information_list.get_mean_service_time_of_types()[0][2]))
    # print("part5:", v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4] + v_action.get_sensing_frequencies()[1] * information_list.get_mean_service_time_of_types()[0][8] + v_action.get_sensing_frequencies()[0] * information_list.get_mean_service_time_of_types()[0][2])
    assert pytest.approx(saq.get_queuing_times()[0]) == pytest.approx((1 / (1 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4] - v_action.get_sensing_frequencies()[1] * information_list.get_mean_service_time_of_types()[0][8])) * \
         (information_list.get_mean_service_time_of_types()[0][2] + ((v_action.get_sensing_frequencies()[2] * information_list.get_second_moment_service_time_of_types()[0][4] + v_action.get_sensing_frequencies()[1] * information_list.get_second_moment_service_time_of_types()[0][8] + v_action.get_sensing_frequencies()[0] * information_list.get_second_moment_service_time_of_types()[0][2]) / (2 * (1 - v_action.get_sensing_frequencies()[2] * information_list.get_mean_service_time_of_types()[0][4] - v_action.get_sensing_frequencies()[1] * information_list.get_mean_service_time_of_types()[0][8] - v_action.get_sensing_frequencies()[0] * information_list.get_mean_service_time_of_types()[0][2])))) - information_list.get_mean_service_time_of_types()[0][2])

    print("\n")
    print("saq_arrival_moments:", saq.get_arrival_moments())
    print("saq_updating_moments:", saq.get_updating_moments())
    print("saq_queuing_time:", saq.get_queuing_times())


v2i = v2iTransmission(
    vehicle=vehicle_list.get_vehicle(vehicle_index=0),
    vehicle_action=v_action,
    edge=edge_node,
    edge_action=e_action,
    arrival_moments=saq.get_arrival_moments(),
    queuing_times=saq.get_queuing_times(),
    white_gaussian_noise=config.white_gaussian_noise,
    mean_channel_fading_gain=config.mean_channel_fading_gain,
    second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
    path_loss_exponent=config.path_loss_exponent,
    information_list=information_list,
)

@pytest.mark.skip(reason="Passed.")
def test_v2iTransmission():
    print("e_action:", e_action.get_bandwidth_allocation())
    print("transmission time:", v2i.get_transmission_times())
    assert int(np.floor(saq.get_arrival_moments()[2] + saq.get_queuing_times()[2])) == 7
    print("data size:", information_list.get_information_siez_by_type(4))
    print("distance:", vehicle_list.get_vehicle(vehicle_index=0).get_distance_between_edge(7, edge_node.get_edge_location()))
    print("SNR:", compute_SNR(
        white_gaussian_noise=config.white_gaussian_noise,
        channel_fading_gain=config.mean_channel_fading_gain,
        distance=vehicle_list.get_vehicle(vehicle_index=0).get_distance_between_edge(7, edge_node.get_edge_location()),
        path_loss_exponent=config.path_loss_exponent,
        transmission_power=v_action.get_transmission_power()))
    print("transmission rate:", compute_transmission_rate(
        SNR=compute_SNR(
            white_gaussian_noise=config.white_gaussian_noise,
            channel_fading_gain=config.mean_channel_fading_gain,
            distance=vehicle_list.get_vehicle(vehicle_index=0).get_distance_between_edge(7, edge_node.get_edge_location()),
            path_loss_exponent=config.path_loss_exponent,
            transmission_power=v_action.get_transmission_power()),
        bandwidth=e_action.get_bandwidth_allocation()[0],
    ))
    assert v2i.get_transmission_times()[2] == (information_list.get_information_siez_by_type(4) / compute_transmission_rate(
        SNR=compute_SNR(
            white_gaussian_noise=config.white_gaussian_noise,
            channel_fading_gain=config.mean_channel_fading_gain,
            distance=vehicle_list.get_vehicle(vehicle_index=0).get_distance_between_edge(7, edge_node.get_edge_location()),
            path_loss_exponent=config.path_loss_exponent,
            transmission_power=v_action.get_transmission_power()),
        bandwidth=e_action.get_bandwidth_allocation()[0],
    ))


