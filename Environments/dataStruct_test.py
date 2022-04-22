import pytest
from Test.environmentConfig_test import vehicularNetworkEnvConfig
from Environments.dataStruct import applicationList, edge, informationList, informationRequirements, softmax, timeSlots, informationPacket, location, trajectory, vehicle, vehicleList, viewList

config = vehicularNetworkEnvConfig()

@pytest.mark.skip(reason="Passed.")
@pytest.mark.parametrize("list_a", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_softmax(list_a):
    assert sum(softmax(list_a)) > 1 - 1e-6 and sum(softmax(list_a)) < 1 + 1e-6

time_slots = timeSlots(
    start=config.time_slot_start, 
    end=config.time_slot_end, 
    slot_length=config.time_slot_length
)

@pytest.mark.skip(reason="Passed.")
def test_timeSlots():
    assert time_slots.is_end() == False
    assert time_slots.get_slot_length() == 1
    assert time_slots.get_number() == 10
    assert time_slots.now() == 0
    time_slots.add_time()
    assert time_slots.now() == 1
    time_slots.reset()
    assert time_slots.now() == 0

information_packet = informationPacket(
    type=1,
    vehicle_index=2,
    edge_index=3,
    updating_moment=4.,
    inter_arrival_interval=5.,
    arrival_moment=6.,
    queuing_time=7.,
    transmission_time=8.,
    received_moment=9.,
)

@pytest.mark.skip(reason="Passed.")
def test_informationPacket():
    assert information_packet.get_type() == 1
    assert information_packet.get_vehicle_index() == 2
    assert information_packet.get_edge_index() == 3
    assert information_packet.get_updating_moment() == 4.
    assert information_packet.get_inter_arrival_interval() == 5.
    assert information_packet.get_arrival_moment() == 6.
    assert information_packet.get_queuing_time() == 7.
    assert information_packet.get_transmission_time() == 8.
    assert information_packet.get_received_moment() == 9.




@pytest.mark.skip(reason="Passed.")
def test_trajectory():
    vehicle_trajectory = trajectory(
        timeSlots=time_slots,
        locations=list(location(1, i) for i in range(10)),
    )
    assert vehicle_trajectory.get_location(0).get_x() == 1.
    assert vehicle_trajectory.get_location(0).get_y() == 0.
    assert vehicle_trajectory.get_location(9).get_x() == 1.
    assert vehicle_trajectory.get_location(9).get_y() == 9.

config.vehicle_list_seeds += [i for i in range(config.vehicle_number)]

vehicle_list = vehicleList(
    number=config.vehicle_number,
    time_slots=time_slots,
    trajectories_file_name=config.trajectories_out_file_name,
    information_number=config.information_number,
    sensed_information_number=config.sensed_information_number,
    min_sensing_cost=config.min_sensing_cost,
    max_sensing_cost=config.max_sensing_cost,
    transmission_power=config.transmission_power,
    seeds=config.vehicle_list_seeds,
)

edge_node = edge(
    edge_index=0,
    information_number=config.information_number,
    edge_location=location(config.edge_location_x, config.edge_location_y),
    communication_range=config.communication_range,
    bandwidth=config.bandwidth,
)

@pytest.mark.skip(reason="Passed.")
def test_vehicle_list():
    # for trajectories in vehicle_list.get_vehicle_trajectories():
    #     print(trajectories)

    for vehicle in vehicle_list.get_vehicle_list():
        print(vehicle.get_distance_between_edge(time_slots.get_start(), edge_location=edge_node.get_edge_location()))
        print(vehicle.get_distance_between_edge(time_slots.get_end(), edge_location=edge_node.get_edge_location()))
        print("\n")


application_list = applicationList(
    number=config.application_number,
    view_number=config.view_number,
    views_per_application=config.views_per_application,
    seed=config.application_list_seed,
)

@pytest.mark.skip(reason="Passed.")
def test_application_list():
    print(application_list.get_application_list())
    assert application_list.get_view_by_application_index(0) == 2
    assert application_list.get_view_by_application_index(config.view_number-1) == 12


config.view_list_seeds += [i for i in range(config.view_number)]
view_list = viewList(
    number=config.view_number,
    information_number=config.information_number,
    required_information_number=config.required_information_number,
    seeds=config.view_list_seeds,
)

@pytest.mark.skip(reason="Passed.")
def test_view_list():
    print(view_list.get_view_list())
    print(view_list.get_information_required_by_view_index(index=0))


information_list = informationList(
    number=config.information_number,
    seed=config.information_list_seed,
    data_size_low_bound=config.data_size_low_bound,
    data_size_up_bound=config.data_size_up_bound,
    data_types_number=config.data_types_number,
    update_interval_low_bound=config.update_interval_low_bound,
    update_interval_up_bound=config.update_interval_up_bound,
    vehicle_list=vehicle_list,
    edge_node=edge_node,
    additive_white_gaussian_noise=config.white_gaussian_noise,
    mean_channel_fading_gain=config.mean_channel_fading_gain,
    second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
    path_loss_exponent=config.path_loss_exponent,
)

def test_information_list():
    print(information_list.get_mean_service_time_of_types())
    print(information_list.get_second_moment_service_time_of_types())

# information_requirements = informationRequirements(
#     time_slots=time_slots,
#     max_application_number=config.max_application_number,
#     min_application_number=config.min_application_number,
#     application_list=application_list,
#     view_list=view_list,
#     information_list=information_list,
#     seed=config.information_requirements_seed,
# )






