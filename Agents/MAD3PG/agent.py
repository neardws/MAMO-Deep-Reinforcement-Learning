"""D3PG agent implementation."""

import copy
import functools
import dataclasses
from typing import Iterator, List, Optional, Tuple, Union, Sequence
from acme import adders
from acme import core
from Environments.environment import vehicularNetworkEnv
from acme import datasets
from Environments import specs
from acme import types
from acme.adders import reverb as reverb_adders
from Agents.MAD3PG import actors
from acme.agents import agent
from Agents.MAD3PG import learning
from acme.tf import networks as network_utils
from acme.tf import utils
from acme.tf import variable_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf

Replicator = Union[snt.distribute.Replicator, snt.distribute.TpuReplicator]

@dataclasses.dataclass
class D3PGConfig:
    """Configuration options for the MAD3PG agent.
    Args:
        environment: Environment specification.
        environment_spec: description of the actions, observations, etc.
        policy_network: the online (optimized) policy.
        critic_network: the online critic.
        observation_network: optional network to transform the observations before
            they are fed into any network.
        discount: discount to use for TD updates.
        batch_size: batch size for updates.
        prefetch_size: size to prefetch from replay.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        policy_optimizer: optimizer for the policy network updates.
        critic_optimizer: optimizer for the critic network updates.
        min_replay_size: minimum replay size before updating.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        n_step: number of steps to squash into a single transition.
        sigma: standard deviation of zero-mean, Gaussian exploration noise.
        clipping: whether to clip gradients by global norm.
        replay_table_name: string indicating what name to give the replay table.
        counter: counter object used to keep track of steps.
        logger: logger object to be used by learner.
        checkpoint: boolean indicating whether to checkpoint the learner.
        accelerator: 'TPU', 'GPU', or 'CPU'. If omitted, the first available accelerator type from ['TPU', 'GPU', 'CPU'] will be selected.
    """
    environment: Optional[vehicularNetworkEnv] = None
    environment_spec: Optional[specs.EnvironmentSpec] = None
    vehicle_policy_network: Optional[snt.Module] = None
    vehicle_critic_network: Optional[snt.Module] = None
    vehicle_observation_network: types.TensorTransformation = tf.identity
    edge_policy_network: Optional[snt.Module] = None
    edge_critic_network: Optional[snt.Module] = None
    edge_observation_network: types.TensorTransformation = tf.identity
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    vehicle_policy_optimizer: Optional[snt.Optimizer] = None
    vehicle_critic_optimizer: Optional[snt.Optimizer] = None
    edge_policy_optimizer: Optional[snt.Optimizer] = None
    edge_critic_optimizer: Optional[snt.Optimizer] = None
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    n_step: int = 5
    sigma: float = 0.3
    clipping: bool = True
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    counter: Optional[counting.Counter] = None
    logger: Optional[loggers.Logger] = None
    checkpoint: bool = True
    accelerator: Optional[str] = None


@dataclasses.dataclass
class D3PGNetworks:
    """Structure containing the networks for D3PG."""

    vehicle_policy_network: types.TensorTransformation
    vehicle_critic_network: types.TensorTransformation
    vehicle_observation_network: types.TensorTransformation

    edge_policy_network: types.TensorTransformation
    edge_critic_network: types.TensorTransformation
    edge_observation_network: types.TensorTransformation

    def __init__(
        self,
        vehicle_policy_network: types.TensorTransformation,
        vehicle_critic_network: types.TensorTransformation,
        vehicle_observation_network: types.TensorTransformation,

        edge_policy_network: types.TensorTransformation,
        edge_critic_network: types.TensorTransformation,
        edge_observation_network: types.TensorTransformation,
    ):
        # This method is implemented (rather than added by the dataclass decorator)
        # in order to allow observation network to be passed as an arbitrary tensor
        # transformation rather than as a snt Module.
        self.vehicle_policy_network = vehicle_policy_network
        self.vehicle_critic_network = vehicle_critic_network
        self.vehicle_observation_network = utils.to_sonnet_module(vehicle_observation_network)

        self.edge_policy_network = edge_policy_network
        self.edge_critic_network = edge_critic_network
        self.edge_observation_network = utils.to_sonnet_module(edge_observation_network)

    def init(
        self, 
        environment_spec: specs.EnvironmentSpec,
    ):
        """Initialize the networks given an environment spec."""
        # Get observation and action specs.
        vehicle_observation_spec = environment_spec.vehicle_observations
        critic_vehicle_action_spec = environment_spec.critic_vehicle_actions
        edge_observation_spec = environment_spec.edge_observations
        critic_edge_action_spec = environment_spec.critic_edge_actions

        # Create variables for the observation net and, as a side-effect, get a
        # spec describing the embedding space.
        vehicle_emb_spec = utils.create_variables(self.vehicle_observation_network, [vehicle_observation_spec])
        edge_emb_spec = utils.create_variables(self.edge_observation_network, [edge_observation_spec])

        # Create variables for the policy and critic nets.
        _ = utils.create_variables(self.vehicle_policy_network, [vehicle_emb_spec])
        _ = utils.create_variables(self.vehicle_critic_network, [vehicle_emb_spec, critic_vehicle_action_spec])

        _ = utils.create_variables(self.edge_policy_network, [edge_emb_spec])
        _ = utils.create_variables(self.edge_critic_network, [edge_emb_spec, critic_edge_action_spec])

    def make_policy(
        self,
        environment_spec: specs.EnvironmentSpec,
        sigma: float = 0.0,
    ) -> Tuple[snt.Module, snt.Module]:
        """Create a single network which evaluates the policy."""
        # Stack the observation and policy networks.
        vehicle_stack = [
            self.vehicle_observation_network,
            self.vehicle_policy_network,
        ]

        edge_stack = [
            self.edge_observation_network,
            self.edge_policy_network,
        ]

        # If a stochastic/non-greedy policy is requested, add Gaussian noise on
        # top to enable a simple form of exploration.
        # TODO: Refactor this to remove it from the class.
        if sigma > 0.0:
            vehicle_stack += [
                network_utils.ClippedGaussian(sigma),
                network_utils.ClipToSpec(environment_spec.vehicle_actions),
            ]
            edge_stack += [
                network_utils.ClippedGaussian(sigma),
                network_utils.ClipToSpec(environment_spec.edge_actions),
            ]

        # Return a network which sequentially evaluates everything in the stack.
        return snt.Sequential(vehicle_stack), snt.Sequential(edge_stack)


class D3PGAgent(agent.Agent):
    """D3PG Agent.
    This implements a single-process D3PG agent. This is an actor-critic algorithm
    that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policy (and as a result the
    behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        config: D3PGConfig,
    ):
        """Initialize the agent.
        Args:
            config: Configuration for the agent.
        """
        self._config = config
        self._accelerator = config.accelerator

        if not self._accelerator:
            self._accelerator = _get_first_available_accelerator_type(['TPU', 'GPU', 'CPU'])

        online_networks = D3PGNetworks(
            vehicle_policy_network=self.config.vehicle_policy_network,
            vehicle_critic_network=self.config.vehicle_critic_network,
            vehicle_observation_network=self.config.vehicle_observation_network,
            edge_policy_network=self.config.edge_policy_network,
            edge_critic_network=self.config.edge_critic_network,
            edge_observation_network=self.config.edge_observation_network,
        )

        # Target networks are just a copy of the online networks.
        target_networks = copy.deepcopy(online_networks)

        # Initialize the networks.
        online_networks.init(self._config.environment_spec)
        target_networks.init(self._config.environment_spec)

        # Create the behavior policy.
        vehicle_policy_network, edge_policy_network = online_networks.make_policy(self._config.environment_spec, self._config.sigma)

        # Create the replay server and grab its address.
        replay_tables = self.make_replay_tables(self._config.environment_spec)
        replay_server = reverb.Server(replay_tables, port=None)
        replay_client = reverb.Client(f'localhost:{replay_server.port}')

        # Create actor, dataset, and learner for generating, storing, and consuming
        # data respectively.
        adder = self.make_adder(replay_client)
        actor = self.make_actor(vehicle_policy_network, edge_policy_network, self._config.environment, adder)
        dataset = self.make_dataset_iterator(replay_client)
        learner = self.make_learner(online_networks, target_networks, dataset, self._config.counter, self._config.logger, self._config.checkpoint)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(self._config.batch_size, self._config.min_replay_size),
            observations_per_step=float(self._config.batch_size) / self._config.samples_per_insert)

        # Save the replay so we don't garbage collect it.
        self._replay_server = replay_server

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer)

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(
                environment_spec))

        return [replay_table]

    def make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size)

        replicator = get_replicator(self._config.accelerator)
        dataset = replicator.experimental_distribute_dataset(dataset)

        # TODO: Fix type stubs and remove.
        return iter(dataset)  # pytype: disable=wrong-arg-types

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount)

    def make_actor(
        self,
        vehicle_policy_network: snt.Module,
        edge_policy_network: snt.Module,
        environment: vehicularNetworkEnv,
        adder: Optional[adders.Adder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ):
        """Create an actor instance."""
        if variable_source:
            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={'vehicle_policy': vehicle_policy_network.variables,
                            'edge_policy': edge_policy_network.variables},
                update_period=1000,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        else:
            variable_client = None

        # Create the actor which defines how we take actions.
        return actors.FeedForwardActor(
            vehicle_policy_network=vehicle_policy_network,
            edge_policy_network=edge_policy_network,
            environment=environment,
            adder=adder,
            variable_client=variable_client,
        )

    def make_learner(
        self,
        online_networks: D3PGNetworks, 
        target_networks: D3PGNetworks,
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = False,
    ):
        """Creates an instance of the learner."""
        # The learner updates the parameters (and initializes them).
        return learning.D3PGLearner(
            vehicle_policy_network=online_networks.vehicle_policy_network,
            vehicle_critic_network=online_networks.vehicle_critic_network,
            edge_policy_network=online_networks.edge_policy_network,
            edge_critic_network=online_networks.edge_critic_network,

            target_vehicle_policy_network=target_networks.vehicle_policy_network,
            target_vehicle_critic_network=target_networks.vehicle_critic_network,
            target_edge_policy_network=target_networks.edge_policy_network,
            target_edge_critic_network=target_networks.edge_critic_network,
            
            discount=self._config.discount,
            target_update_period=self._config.target_update_period,
            dataset_iterator=dataset,

            vehicle_observation_network=online_networks.vehicle_observation_network,
            target_vehicle_observation_network=target_networks.vehicle_observation_network,
            edge_observation_network=online_networks.edge_observation_network,
            target_edge_observation_network=target_networks.edge_observation_network,

            vehicle_policy_optimizer=self._config.vehicle_policy_optimizer,
            vehicle_critic_optimizer=self._config.vehicle_critic_optimizer,
            edge_policy_optimizer=self._config.edge_policy_optimizer,
            edge_critic_optimizer=self._config.edge_critic_optimizer,

            clipping=self._config.clipping,
            replicator=get_replicator(self._config.accelerator),

            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )

def _ensure_accelerator(accelerator: str) -> str:
    """Checks for the existence of the expected accelerator type.
    Args:
        accelerator: 'CPU', 'GPU' or 'TPU'.
    Returns:
        The validated `accelerator` argument.
    Raises:
        RuntimeError: Thrown if the expected accelerator isn't found.
    """
    devices = tf.config.get_visible_devices(device_type=accelerator)

    if devices:
        return accelerator
    else:
        error_messages = [f'Couldn\'t find any {accelerator} devices.',
                        'tf.config.get_visible_devices() returned:']
        error_messages.extend([str(d) for d in devices])
        raise RuntimeError('\n'.join(error_messages))


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

# Only instantiate one replicator per (process, accelerator type), in case
# a replicator stores state that needs to be carried between its method calls.
@functools.lru_cache()
def get_replicator(accelerator: Optional[str]) -> Replicator:
    """Returns a replicator instance appropriate for the given accelerator.
    This caches the instance using functools.cache, so that only one replicator
    is instantiated per process and argument value.
    Args:
        accelerator: None, 'TPU', 'GPU', or 'CPU'. If None, the first available
        accelerator type will be chosen from ('TPU', 'GPU', 'CPU').
    Returns:
        A replicator, for replciating weights, datasets, and updates across
        one or more accelerators.
    """
    if accelerator:
        accelerator = _ensure_accelerator(accelerator)
    else:
        accelerator = _get_first_available_accelerator_type()

    if accelerator == 'TPU':
        tf.tpu.experimental.initialize_tpu_system()
        return snt.distribute.TpuReplicator()
    else:
        return snt.distribute.Replicator()