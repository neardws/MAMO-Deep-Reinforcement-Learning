import dataclasses
from typing import List

@dataclasses.dataclass
class vehicularNetworkEnvConfig:
    """Configuration for the vehicular network environment."""
    
    """Time slot related."""
    time_slot_start: int = 0
    time_slot_end: int = 299
    time_slot_length: int = 1

    """Information list related."""
    information_number: int = 50
    information_list_seed: int = 0
    data_size_low_bound: float = 100      # Bytes
    data_size_up_bound: float = 1 * 1024 * 1024    # Bytes
    data_types_number: int = 20
    update_interval_low_bound: int = 1
    update_interval_up_bound: int = 10

    """Vehicles related."""
    vehicle_number: int = 10
    trajectories_file_name: str = 'CSV/trajectories_20161116_0800_0850.csv'
    sensed_information_number: int = 10  # the maximum number of information, which can be sensed by the vehicle.
    min_sensing_cost: float = 0.1
    max_sensing_cost: float = 1.0
    transmission_power: float = 100.0  # mW
    vehicle_list_seeds: List[int] = dataclasses.field(default_factory=list)

    """"Edge related."""
    edge_index: int = 0
    edge_location_x: float = 500.0   # meters
    edge_location_y: float = 500.0   # meters
    communication_range: float = 500.0  # meters
    bandwidth: float = 3.0  # MHz

    """View related."""
    view_number: int = 30
    required_information_number: int = 10  # the maximume number of information required by one view.
    view_list_seeds: List[int] = dataclasses.field(default_factory=list)

    """Application related."""
    application_number: int = 0
    views_per_application: int = 1
    application_list_seed: int = 0

    """Information Requirements related."""
    max_application_number: int = 3
    min_application_number: int = 1
    information_requirements_seed: int = 0

    """Vehicle Trajectories Processor related."""
    trajectories_file_name: str = 'CSV/gps_20161116'
    longitude_min: float = 104.04565967220308
    latitude_min: float = 30.654605745741608
    map_width: float = 1000.0   # meters
    trajectories_time_start: str = '2016-11-16 08:00:00'
    trajectories_time_end: str = '2016-11-16 08:05:00'
    trajectories_out_file_name: str = 'CSV/trajectories_20161116_0800_0850.csv'

    """V2I Transmission related."""
    white_gaussian_noise: int = -90  # dBm
    mean_channel_fading_gain: float = 2.0 
    second_moment_channel_fading_gain: float = 0.4
    path_loss_exponent: int = 3
    SNR_target_low_bound: float = 30 # dB
    SNR_target_up_bound: float = 35 # dB
    probabiliity_threshold: float = 0.9

    """Age of View related."""
    wight_of_timeliness: float = 0.3
    wight_of_consistency: float = 0.3
    wight_of_redundancy: float = 0.2
    wight_of_cost: float = 0.2