"""D3PG learner implementation."""

import time
from typing import Dict, Iterator, List, Optional, Union, Sequence

import acme
from acme import types
from acme.tf import losses
from acme.tf import networks as acme_nets
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree

from Environments.environment import vehicularNetworkEnv

Replicator = Union[snt.distribute.Replicator, snt.distribute.TpuReplicator]

class D3PGLearner(acme.Learner):
    """MAD3PG learner.

    This is the learning component of a D3PG agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        vehicle_policy_network: snt.Module,
        vehicle_critic_network: snt.Module,
        edge_policy_network: snt.Module,
        edge_critic_network: snt.Module,
        
        target_vehicle_policy_network: snt.Module,
        target_vehicle_critic_network: snt.Module,
        target_edge_policy_network: snt.Module,
        target_edge_critic_network: snt.Module,
        
        discount: float,
        target_update_period: int,
        dataset_iterator: Iterator[reverb.ReplaySample],
        
        vehicle_observation_network: types.TensorTransformation = lambda x: x,
        target_vehicle_observation_network: types.TensorTransformation = lambda x: x,
        edge_observation_network: types.TensorTransformation = lambda x: x,
        target_edge_observation_network: types.TensorTransformation = lambda x: x,
        
        vehicle_policy_optimizer: Optional[snt.Optimizer] = None,
        vehicle_critic_optimizer: Optional[snt.Optimizer] = None,
        edge_policy_optimizer: Optional[snt.Optimizer] = None,
        edge_critic_optimizer: Optional[snt.Optimizer] = None,
        
        clipping: bool = True,
        replicator: Optional[Replicator] = None,

        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        
        environment: Optional[vehicularNetworkEnv] = None,
    ):
        """Initializes the learner.

        Args:
        policy_network: the online (optimized) policy.
        critic_network: the online critic.
        target_policy_network: the target policy (which lags behind the online
            policy).
        target_critic_network: the target critic.
        discount: discount to use for TD updates.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        dataset_iterator: dataset to learn from, whether fixed or from a replay
            buffer (see `acme.datasets.reverb.make_reverb_dataset` documentation).
        observation_network: an optional online network to process observations
            before the policy and the critic.
        target_observation_network: the target observation network.
        policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
        critic_optimizer: the optimizer to be applied to the distributional
            Bellman loss.
        clipping: whether to clip gradients by global norm.
        replicator: Replicates variables and their update methods over multiple
        accelerators, such as the multiple chips in a TPU.
        counter: counter object used to keep track of steps.
        logger: logger object to be used by learner.
        checkpoint: boolean indicating whether to checkpoint the learner.
        """

        # Store online and target networks.
        self._vehicle_policy_network = vehicle_policy_network
        self._vehicle_critic_network = vehicle_critic_network
        self._edge_policy_network = edge_policy_network
        self._edge_critic_network = edge_critic_network

        self._target_vehicle_policy_network = target_vehicle_policy_network
        self._target_vehicle_critic_network = target_vehicle_critic_network
        self._target_edge_policy_network = target_edge_policy_network
        self._target_edge_critic_network = target_edge_critic_network

        # Make sure observation networks are snt.Module's so they have variables.
        self._vehicle_observation_network = tf2_utils.to_sonnet_module(vehicle_observation_network)
        self._target_vehicle_observation_network = tf2_utils.to_sonnet_module(target_vehicle_observation_network)
        self._edge_observation_network = tf2_utils.to_sonnet_module(edge_observation_network)
        self._target_edge_observation_network = tf2_utils.to_sonnet_module(target_edge_observation_network)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger('learner')

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Replicates Variables across multiple accelerators
        if not replicator:
            accelerator = _get_first_available_accelerator_type()
            if accelerator == 'TPU':
                replicator = snt.distribute.TpuReplicator()
            else:
                replicator = snt.distribute.Replicator()

        self._replicator = replicator

        with replicator.scope():
            # Necessary to track when to update target networks.
            self._num_steps = tf.Variable(0, dtype=tf.int32)
            self._target_update_period = target_update_period

            # Create optimizers if they aren't given.
            self._vehicle_policy_optimizer = vehicle_policy_optimizer or snt.optimizers.Adam(1e-4)
            self._vehicle_critic_optimizer = vehicle_critic_optimizer or snt.optimizers.Adam(1e-4)

            self._edge_policy_optimizer = edge_policy_optimizer or snt.optimizers.Adam(1e-4)
            self._edge_critic_optimizer = edge_critic_optimizer or snt.optimizers.Adam(1e-4)

        # Batch dataset and create iterator.
        self._iterator = dataset_iterator

        # Expose the variables.
        vehicle_policy_network_to_expose = snt.Sequential(
            [self._target_vehicel_observation_network, self._target_vehicle_policy_network])
        edge_policy_network_to_expose = snt.Sequential(
            [self._target_edge_observation_network, self._target_edge_policy_network])
        self._variables = {
            'vehicle_critic': self._target_vehicle_critic_network.variables,
            'vehicle_policy': vehicle_policy_network_to_expose.variables,
            'edge_critic': self._target_edge_critic_network.variables,
            'edge_policy': edge_policy_network_to_expose.variables,
        }


        # Create a checkpointer and snapshotter objects.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                subdirectory='d3pg_learner',
                objects_to_save={
                    'counter': self._counter,

                    'vehicle_policy': self._vehicle_policy_network,
                    'vehicle_critic': self._vehicle_critic_network,
                    'vehicle_observation': self._vehicle_observation_network,
                    'target_vehicle_policy': self._target_vehicle_policy_network,
                    'target_vehicle_critic': self._target_vehicle_critic_network,
                    'target_vehicle_observation': self._target_vehicle_observation_network,
                    'vehicle_policy_optimizer': self._vehicle_policy_optimizer,
                    'vehicle_critic_optimizer': self._vehicle_critic_optimizer,

                    'edge_policy': self._edge_policy_network,
                    'edge_critic': self._edge_critic_network,
                    'edge_observation': self._edge_observation_network,
                    'target_edge_policy': self._target_edge_policy_network,
                    'target_edge_critic': self._target_edge_critic_network,
                    'target_edge_observation': self._target_edge_observation_network,
                    'edge_policy_optimizer': self._edge_policy_optimizer,
                    'edge_critic_optimizer': self._edge_critic_optimizer,

                    'num_steps': self._num_steps,
                })
            vehicle_critic_mean = snt.Sequential(
                [self._vehicle_critic_network, acme_nets.StochasticMeanHead()])
            edge_critic_mean = snt.Sequential(
                [self._edge_critic_network, acme_nets.StochasticMeanHead()])
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={
                    'vehicle_policy': self._vehicle_policy_network,
                    'vehicle_critic': vehicle_critic_mean,
                    'edge_policy': self._edge_policy_network,
                    'edge_critic': edge_critic_mean,
                })

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

        self._environment = environment

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        # Update target network
        online_variables = (
            *self._vehicle_observation_network.variables,
            *self._vehicle_critic_network.variables,
            *self._vehicle_policy_network.variables,
            *self._edge_observation_network.variables,
            *self._edge_critic_network.variables,
            *self._edge_policy_network.variables,
        )
        target_variables = (
            *self._target_vehicle_observation_network.variables,
            *self._target_vehicle_critic_network.variables,
            *self._target_vehicle_policy_network.variables,
            *self._target_edge_observation_network.variables,
            *self._target_edge_critic_network.variables,
            *self._target_edge_policy_network.variables,
        )

        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)
        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        sample = next(self._iterator)
        transitions: types.Transition = sample.data  # Assuming ReverbSample.

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

        with tf.GradientTape(persistent=True) as tape:
            """Deal with the observations."""
            vehicle_observations_list: List[List[np.ndarray]] = []
            for _ in range(self._environment._config.vehicle_number):
                vehicle_observations_list.append([])

            vehicle_next_observations_list: List[List[np.ndarray]] = []
            for _ in range(self._environment._config.vehicle_number):
                vehicle_next_observations_list.append([])

            for observation in transitions.observation:
                vehicle_observations = vehicularNetworkEnv.get_vehicle_observations(
                    vehicle_number=self._environment.vehicle_number, 
                    information_number=self._environment.information_number, 
                    sensed_information_number=self._environment.sensed_information_number, 
                    vehicle_observation_size=self._environment.vehicle_observation_size,
                    observation=observation,
                )
                for i, vehicle_observation in enumerate(vehicle_observations):
                    vehicle_observations_list[i].append(vehicle_observation)
            
            for observation in transitions.next_observation:
                vehicle_observations = vehicularNetworkEnv.get_vehicle_observations(
                    vehicle_number=self._environment.vehicle_number, 
                    information_number=self._environment.information_number, 
                    sensed_information_number=self._environment.sensed_information_number, 
                    vehicle_observation_size=self._environment.vehicle_observation_size,
                    observation=observation,
                )
                for i, vehicle_observation in enumerate(vehicle_observations):
                    vehicle_next_observations_list[i].append(vehicle_observation)

            vehicle_observations_np_array_list = []
            for vehicle_index in range(self._environment._config.vehicle_number):
                vehicle_observations_np_array: np.ndarray = np.expand_dims(vehicle_observations_list[vehicle_index][0], axis=0)
                for observation in vehicle_observations_list[vehicle_index][1:]:
                    vehicle_observations_np_array = np.concatenate((vehicle_observations_np_array, np.expand_dims(observation, axis=0)), axis=0)
                vehicle_observations_np_array_list.append(vehicle_observations_np_array)

            vehicle_next_observations_np_array_list = []
            for vehicle_index in range(self._environment._config.vehicle_number):
                vehicle_next_observations_np_array: np.ndarray = np.expand_dims(vehicle_next_observations_list[vehicle_index][0], axis=0)
                for observation in vehicle_next_observations_list[vehicle_index][1:]:
                    vehicle_next_observations_np_array = np.concatenate((vehicle_next_observations_np_array, np.expand_dims(observation, axis=0)), axis=0)
                vehicle_next_observations_np_array_list.append(vehicle_next_observations_np_array)
            
            # TODO: Use tf.stack instead of tf.concat.
            vehicles_a_t_list = []
            o_t = self._target_vehicle_observation_network(vehicle_next_observations_np_array_list[vehicle_index])
            o_t = tree.map_structure(tf.stop_gradient, o_t)
            vehicles_a_t = self._target_vehicle_policy_network(o_t)
            vehicles_a_t_list.append(vehicles_a_t)
            for vehicle_index in range(1, self._environment._config.vehicle_number):
                o_t = self._target_vehicle_observation_network(vehicle_next_observations_np_array_list[vehicle_index])
                o_t = tree.map_structure(tf.stop_gradient, o_t)
                a_t = self._target_vehicle_policy_network(o_t)
                vehicles_a_t_list.append(a_t)
                vehicles_a_t = tf.concat([vehicles_a_t, a_t], axis=0)

            """Compute the loss for the policy and critic of vehicles."""
            vehicle_critic_losses = []
            vehicle_policy_losses = []
            for vehicle_index in range(self._environment._config.vehicle_number):
                # Maybe transform the observation before feeding into policy and critic.
                # Transforming the observations this way at the start of the learning
                # step effectively means that the policy and critic share observation
                # network weights.

                o_tm1 = self._vehicle_observation_network(vehicle_observations_np_array_list[vehicle_index])
                o_t = self._target_vehicle_observation_network(vehicle_next_observations_np_array_list[vehicle_index])
                # This stop_gradient prevents gradients to propagate into the target
                # observation network. In addition, since the online policy network is
                # evaluated at o_t, this also means the policy loss does not influence
                # the observation network training.
                o_t = tree.map_structure(tf.stop_gradient, o_t)

                # Critic learning.
                q_tm1 = self._vehicle_critic_network(o_tm1, transitions.action[: self._environment._config.vehicle_number * self._environment._vehicle_action_size])
                q_t = self._target_vehicle_critic_network(o_t, vehicles_a_t)

                # Critic loss.
                vehicle_critic_loss = losses.categorical(q_tm1, transitions.reward[vehicle_index],
                                                discount * transitions.discount, q_t)
                vehicle_critic_loss = tf.reduce_mean(vehicle_critic_loss, axis=[0])
                vehicle_critic_losses.append(vehicle_critic_loss)

                # Actor learning
                if vehicle_index == 0:
                    dpg_a_t = self._vehicle_policy_network(o_t)
                else:
                    dpg_a_t = vehicles_a_t_list[0]
                for i in range(self._environment._config.vehicle_number):
                    if i != 0 and i != vehicle_index:
                        dpg_a_t = tf.concat([dpg_a_t, vehicles_a_t_list[i]], axis=0)
                    elif i != 0 and i == vehicle_index:
                        dpg_a_t = tf.concat([dpg_a_t, self._vehicle_policy_network(o_t)], axis=0)
                
                dpg_z_t = self._vehicle_critic_network(o_t, dpg_a_t)
                dpg_q_t = dpg_z_t.mean()

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._clipping else None
                vehicle_policy_loss = losses.dpg(
                    dpg_q_t,
                    vehicles_a_t,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping)
                vehicle_policy_loss = tf.reduce_mean(vehicle_policy_loss, axis=[0])
                vehicle_policy_losses.append(vehicle_policy_loss)
            
            """Compute the mean loss for the policy and critic of vehicles."""
            vehicle_critic_loss = tf.reduce_mean(tf.stack(vehicle_critic_losses, axis=0), axis=0)
            vehicle_policy_loss = tf.reduce_mean(tf.stack(vehicle_policy_losses, axis=0), axis=0)

            o_tm1 = self._edge_observation_network(transitions.observation)
            o_t = self._target_edge_observation_network(transitions.next_observation)
            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t = tree.map_structure(tf.stop_gradient, o_t)

            # Critic learning.
            a_t = self._target_edge_policy_network(o_t)
            a_t = tf.concat([vehicles_a_t, a_t], axis=0)
            q_tm1 = self._edge_critic_network(o_tm1, transitions.action)
            q_t = self._target_edge_critic_network(o_t, a_t)

            # Critic loss.
            edge_critic_loss = losses.categorical(q_tm1, transitions.reward[-1],
                                            discount * transitions.discount, q_t)
            edge_critic_loss = tf.reduce_mean(edge_critic_loss, axis=[0])

            # Actor learning.
            dpg_a_t = self._edge_policy_network(o_t)
            dpg_a_t = tf.concat([vehicles_a_t, dpg_a_t], axis=0)
            dpg_z_t = self._edge_critic_network(o_t, dpg_a_t)
            dpg_q_t = dpg_z_t.mean()

            # Actor loss. If clipping is true use dqda clipping and clip the norm.
            dqda_clipping = 1.0 if self._clipping else None
            edge_policy_loss = losses.dpg(
                dpg_q_t,
                dpg_a_t,
                tape=tape,
                dqda_clipping=dqda_clipping,
                clip_norm=self._clipping)
            edge_policy_loss = tf.reduce_mean(edge_policy_loss, axis=[0])

        # Get trainable variables.
        vehicle_policy_variables = self._vehicle_policy_network.trainable_variables
        vehicle_critic_variables = (
            # In this agent, the critic loss trains the observation network.
            self._vehicle_observation_network.trainable_variables +
            self._vehicle_critic_network.trainable_variables)

        edge_policy_variables = self._edge_policy_network.trainable_variables
        edge_critic_variables = (
            # In this agent, the critic loss trains the observation network.
            self._edge_observation_network.trainable_variables +
            self._edge_critic_network.trainable_variables)

        # Compute gradients.
        replica_context = tf.distribute.get_replica_context()
        vehicle_policy_gradients =  _average_gradients_across_replicas(
            replica_context,
            tape.gradient(vehicle_policy_loss, vehicle_policy_variables))
        vehicle_critic_gradients =  _average_gradients_across_replicas(
            replica_context,
            tape.gradient(vehicle_critic_loss, vehicle_critic_variables))
        edge_policy_gradients =  _average_gradients_across_replicas(
            replica_context,
            tape.gradient(edge_policy_loss, edge_policy_variables))
        edge_critic_gradients =  _average_gradients_across_replicas(
            replica_context,
            tape.gradient(edge_critic_loss, edge_critic_variables))

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            vehicle_policy_gradients = tf.clip_by_global_norm(vehicle_policy_gradients, 40.)[0]
            vehicle_critic_gradients = tf.clip_by_global_norm(vehicle_critic_gradients, 40.)[0]
            edge_policy_gradients = tf.clip_by_global_norm(edge_policy_gradients, 40.)[0]
            edge_critic_gradients = tf.clip_by_global_norm(edge_critic_gradients, 40.)[0]

        # Apply gradients.
        self._vehicle_policy_optimizer.apply(vehicle_policy_gradients, vehicle_policy_variables)
        self._vehicle_critic_optimizer.apply(vehicle_critic_gradients, vehicle_critic_variables)
        self._edge_policy_optimizer.apply(edge_policy_gradients, edge_policy_variables)
        self._edge_critic_optimizer.apply(edge_critic_gradients, edge_critic_variables)

        # Losses to track.
        return {
            'vehicle_policy_loss': vehicle_policy_loss,
            'vehicle_critic_loss': vehicle_critic_loss,
            'edge_policy_loss': edge_policy_loss,
            'edge_critic_loss': edge_critic_loss,
        }

    @tf.function
    def _replicated_step(self):
        # Update target network
        online_variables = (
            *self._vehicle_observation_network.variables,
            *self._vehicle_critic_network.variables,
            *self._vehicle_policy_network.variables,
            *self._edge_observation_network.variables,
            *self._edge_critic_network.variables,
            *self._edge_policy_network.variables,
        )
        target_variables = (
            *self._target_vehicle_observation_network.variables,
            *self._target_vehicle_critic_network.variables,
            *self._target_vehicle_policy_network.variables,
            *self._target_edge_observation_network.variables,
            *self._target_edge_critic_network.variables,
            *self._target_edge_policy_network.variables,
        )

        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)
        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        sample = next(self._iterator)

        # This mirrors the structure of the fetches returned by self._step(),
        # but the Tensors are replaced with replicated Tensors, one per accelerator.
        replicated_fetches = self._replicator.run(self._step, args=(sample,))

        def reduce_mean_over_replicas(replicated_value):
            """Averages a replicated_value across replicas."""
            # The "axis=None" arg means reduce across replicas, not internal axes.
            return self._replicator.reduce(
                reduce_op=tf.distribute.ReduceOp.MEAN,
                value=replicated_value,
                axis=None)

        fetches = tree.map_structure(reduce_mean_over_replicas, replicated_fetches)

        return fetches

    def step(self):
        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
            self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]

def _get_first_available_accelerator_type(
    wishlist: Sequence[str] = ('TPU', 'GPU', 'CPU')) -> str:
    """Returns the first available accelerator type listed in a wishlist.
    Args:
        wishlist: A sequence of elements from {'CPU', 'GPU', 'TPU'}, listed in
        order of descending preference.
    Returns:
        The first available accelerator type from `wishlist`.
    Raises:
        RuntimeError: Thrown if no accelerators from the `wishlist` are found.
    """
    get_visible_devices = tf.config.get_visible_devices

    for wishlist_device in wishlist:
        devices = get_visible_devices(device_type=wishlist_device)
        if devices:
            return wishlist_device

    available = ', '.join(
        sorted(frozenset([d.type for d in get_visible_devices()])))
    raise RuntimeError(
        'Couldn\'t find any devices from {wishlist}.' +
        f'Only the following types are available: {available}.')


def _average_gradients_across_replicas(replica_context, gradients):
    """Computes the average gradient across replicas.
    This computes the gradient locally on this device, then copies over the
    gradients computed on the other replicas, and takes the average across
    replicas.
    This is faster than copying the gradients from TPU to CPU, and averaging
    them on the CPU (which is what we do for the losses/fetches).
    Args:
        replica_context: the return value of `tf.distribute.get_replica_context()`.
        gradients: The output of tape.gradients(loss, variables)
    Returns:
        A list of (d_loss/d_varabiable)s.
    """

    # We must remove any Nones from gradients before passing them to all_reduce.
    # Nones occur when you call tape.gradient(loss, variables) with some
    # variables that don't affect the loss.
    # See: https://github.com/tensorflow/tensorflow/issues/783
    gradients_without_nones = [g for g in gradients if g is not None]
    original_indices = [i for i, g in enumerate(gradients) if g is not None]

    results_without_nones = replica_context.all_reduce('mean',
                                                        gradients_without_nones)
    results = [None] * len(gradients)
    for ii, result in zip(original_indices, results_without_nones):
        results[ii] = result

    return results 