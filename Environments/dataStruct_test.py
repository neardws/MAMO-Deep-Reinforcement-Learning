import pytest
import numpy as np
from Test.environmentConfig_test import vehicularNetworkEnvConfig
from Environments.dataStruct import applicationList, edge, informationList, softmax, timeSlots, informationPacket, location, trajectory, vehicleAction, vehicleList, viewList
from Environments.dataStruct import informationRequirements, edgeAction

config = vehicularNetworkEnvConfig()

@pytest.mark.skip(reason="Passed.")
def test_softmax():
    assert sum(softmax([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) > 1 - 1e-6 and sum(softmax([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) < 1 + 1e-6

time_slots = timeSlots(
    start=config.time_slot_start, 
    end=config.time_slot_end, 
    slot_length=config.time_slot_length
)

@pytest.mark.skip(reason="Passed.")
def test_timeSlots():
    assert time_slots.is_end() == False
    assert time_slots.get_slot_length() == 1
    assert time_slots.get_number() == 300
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
        locations=list(location(1, i) for i in range(300)),
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

@pytest.mark.skip(reason="Passed.")
def test_vehicle():
    v = vehicle_list.get_vehicle_list()[0]
    print(v.get_information_canbe_sensed())
    print(v.get_sensed_information_type(sensed_information=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))



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
    white_gaussian_noise=config.white_gaussian_noise,
    mean_channel_fading_gain=config.mean_channel_fading_gain,
    second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
    path_loss_exponent=config.path_loss_exponent,
)

@pytest.mark.skip(reason="Passed.")
def test_information_list():
    print(information_list.get_mean_service_time_of_types())
    print(information_list.get_second_moment_service_time_of_types())
    print(information_list.get_mean_service_time_by_vehicle_and_type(vehicle_index=0, data_type_index=0))

vehicle_action = vehicleAction(
    vehicle_index=0,
    now_time=0,
    sensed_information=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    sensing_frequencies=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    uploading_priorities=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    transmission_power=0.3, 
    action_time=0
)

# notice that vehicle.get_sensed_information_number() is not equal to len(self._sensed_information)
@pytest.mark.skip(reason="Passed.")
def test_vehicle_action():
    assert vehicle_action._action_time == 0
    assert vehicle_action._vehicle_index <= len(vehicle_list.get_vehicle_list())
    vehicle = vehicle_list.get_vehicle(vehicle_action._vehicle_index)
    assert len(vehicle_action._sensed_information) == len(vehicle_action._sensing_frequencies) == len(vehicle_action._uploading_priorities)
    assert vehicle_action._transmission_power <= vehicle.get_transmission_power()
    assert vehicle_action.check_action(nowTimeSlot=0, vehicle_list=vehicle_list) == True

np.random.seed(0)
random_action = np.random.random(size=config.sensed_information_number*3+1)
v_action: vehicleAction = vehicleAction.generate_from_np_array(
    now_time=10,
    vehicle_index=0,
    vehicle_list=vehicle_list,
    information_list=information_list,
    sensed_information_number=config.sensed_information_number,
    network_output=random_action,
    white_gaussian_noise=config.white_gaussian_noise,
    mean_channel_fading_gain=config.mean_channel_fading_gain,
    second_moment_channel_fading_gain=config.second_moment_channel_fading_gain,
    edge_location=edge_node.get_edge_location(),
    path_loss_exponent=config.path_loss_exponent,
    SNR_target_low_bound=config.SNR_target_low_bound,
    SNR_target_up_bound=config.SNR_target_up_bound,
    probabiliity_threshold=config.probabiliity_threshold,
    action_time=10
)

@pytest.mark.skip(reason="Passed.")
def test_generate_vehicle_action():
    print("v_action.get_sensed_information():")
    print(v_action.get_sensed_information())
    print("v_action.get_sensing_frequencies():")
    print(v_action.get_sensing_frequencies())
    print("v_action.get_uploading_priorities():")
    print(v_action.get_uploading_priorities())
    print("v_action.get_transmission_power():")
    print(v_action.get_transmission_power())


edge_action = edgeAction(
    edge=edge_node,
    now_time=0,
    vehicle_number=config.vehicle_number,
    bandwidth_allocation=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    action_time=0,
)

@pytest.mark.skip(reason="Passed.")
def test_edge_action():
    assert edge_action.check_action(nowTimeSlot=0) == True
    print(edge_action.get_bandwidth_allocation())


np.random.seed(0)
random_action = np.random.random(size=config.vehicle_number)
# print(random_action)
e_action = edgeAction.generate_from_np_array(
    now_time=0,
    edge_node=edge_node,
    action_time=0,
    network_output=random_action,
    vehicle_number=config.vehicle_number,
)

@pytest.mark.skip(reason="Passed.")
def test_generate_edge_action():
    print("edge_action.get_bandwidth_allocation():")
    print(e_action.get_bandwidth_allocation())
    assert pytest.approx(e_action.get_bandwidth_allocation().sum()) == edge_node.get_bandwidth()

information_requirements = informationRequirements(
    time_slots=time_slots,
    max_application_number=config.max_application_number,
    min_application_number=config.min_application_number,
    application_list=application_list,
    view_list=view_list,
    information_list=information_list,
    seed=config.information_requirements_seed,
)

@pytest.mark.skip(reason="Passed.")
def test_informationRequirements():
    assert (information_requirements.applications_at_now(nowTimeStamp=0)) == [1, 9]
    assert (information_requirements.views_required_by_application_at_now(nowTimeStamp=0)) == [28, 22]
    assert (information_requirements.get_views_required_number_at_now(nowTimeStamp=0)) == 2
    assert (information_requirements.get_information_type_required_by_views_at_now_at_now(nowTimeStamp=0)) == [[5, 4], [4, 8, 5, 7]]
    assert list(information_requirements.get_information_required_at_now(nowTimeStamp=0)) == [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]