import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")
from Environments.utilities import vehicleTrajectoriesProcessor

if __name__ == "__main__":
    # default scenario
    # trajectory_processor = vehicleTrajectoriesProcessor(
    #     file_name='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/gps_20161116', 
    #     longitude_min=104.04565967220308, 
    #     latitude_min=30.654605745741608, 
    #     map_width=1000.0,
    #     time_start='2016-11-16 08:00:00', 
    #     time_end='2016-11-16 08:05:00', 
    #     out_file='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/trajectories_20161116_0800_0850.csv',
    # )
    
    trajectory_processor = vehicleTrajectoriesProcessor(
        file_name='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/gps_20161116', 
        longitude_min=104.04565967220308, 
        latitude_min=30.654605745741608, 
        map_width=1000.0,
        time_start='2016-11-16 23:00:00', 
        time_end='2016-11-16 23:05:00', 
        out_file='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/trajectories_20161116_2300_2305.csv',
    )
    
    trajectory_processor = vehicleTrajectoriesProcessor(
        file_name='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/gps_20161127', 
        longitude_min=108.93445967220308, 
        latitude_min=34.22454605745741, 
        map_width=1000.0,
        time_start='2016-11-27 08:00:00', 
        time_end='2016-11-27 08:05:00', 
        out_file='/home/neardws/Documents/AoV-Journal-Algorithm/CSV/trajectories_20161127_0800_0805.csv',
    )