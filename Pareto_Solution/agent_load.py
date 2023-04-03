from Agents.MAMOD3PG.actors import FeedForwardActor
import tensorflow as tf

base_dir = "/home/neardws/acme/69b15758-5087-11ed-992a-04d9f5632a58/snapshots/"

edge_policy_dir = base_dir + "edge_policy/"
vehicle_policy_dir = base_dir + "vehicle_policy"

edge_policy = tf.saved_model.load(edge_policy_dir)
vehicle_policy = tf.saved_model.load(vehicle_policy_dir)



