import pytest
import numpy as np
from Test.environmentConfig_test import vehicularNetworkEnvConfig
from Environments.utilities import vehicleTrajectoriesProcessor, v2iTransmission
from Environments.utilities import get_minimum_transmission_power, generate_channel_fading_gain
from Environments.utilities import compute_SNR, compute_transmission_rate, compute_successful_tansmission_probability
from Environments.utilities import cover_MHz_to_Hz, cover_ratio_to_dB, cover_dB_to_ratio, cover_dBm_to_W, cover_W_to_dBm, cover_W_to_mW, cover_mW_to_W, cover_bps_to_Mbps
from Log.logger import myapp

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

def test_sensingAndQueuing():
    pass

def test_v2iTransmission():
    pass
