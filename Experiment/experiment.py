import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")
from absl import app
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
memory_limit=4 * 1024
tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
tf.config.experimental.set_virtual_device_configuration(gpus[1], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

from Experiment import run_mad3pg

if __name__ == '__main__':
    app.run(run_mad3pg.main)
