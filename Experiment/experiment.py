import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")
from absl import app
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
memory_limit=8 * 1024
tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
memory_limit=8 * 1024
tf.config.experimental.set_virtual_device_configuration(gpus[1], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
from Experiment import run_ra
from Experiment import run_d3pg
from Experiment import run_d4pg
from Experiment import run_mad3pg
from Experiment import run_mad3pg_dr
from Experiment import run_mamo
from Experiment import run_mamod3pg
from Experiment import run_mamo_for_pareto


if __name__ == '__main__':
    # app.run(run_ra.main)
    # app.run(run_d3pg.main)
    # app.run(run_d4pg.main)
    # app.run(run_mad3pg.main)
    # app.run(run_mad3pg_dr.main)
    # app.run(run_mamod3pg.main)
    app.run(run_mamo.main)
    # app.run(run_mamo_for_pareto.main)
