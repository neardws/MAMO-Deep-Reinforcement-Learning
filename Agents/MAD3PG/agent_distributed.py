"""Defines the MAD3PG agent class."""

import copy
from typing import Optional

import acme
from Environments.environment import vehicularNetworkEnv
from Agents.MAD3PG.agent import D3PGNetworks, D3PGConfig, D3PGAgent
from acme.tf import savers as tf2_savers
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import launchpad as lp
import reverb

# Valid values of the "accelerator" argument.
_ACCELERATORS = ('CPU', 'GPU', 'TPU')

class MultiAgentDistributedDDPG:
    """Program definition for MAD3PG."""
    def __init__(
        self,
        config: D3PGConfig,
        num_actors: int = 1,
        num_caches: int = 0,
        max_actor_steps: Optional[int] = None,
        log_every: float = 10.0,
    ):
        """Initialize the MAD3PG agent."""
        self._config = config

        self._accelerator = config.accelerator
        if self._accelerator is not None and self._accelerator not in _ACCELERATORS:
            raise ValueError(f'Accelerator must be one of {_ACCELERATORS}, '
                            f'not "{self._accelerator}".')

        self._num_actors = num_actors
        self._num_caches = num_caches
        self._max_actor_steps = max_actor_steps
        self._log_every = log_every

        # Create the agent.
        self._agent = D3PGAgent(self._config)

    def replay(self):
        """The replay storage."""
        return self._agent.make_replay_tables(self._config.environment_spec)

    def counter(self):
        return tf2_savers.CheckpointingRunner(counting.Counter(),
                                            time_delta_minutes=1,
                                            subdirectory='counter')

    def coordinator(self, counter: counting.Counter):
        return lp_utils.StepsLimiter(counter, self._max_actor_steps)

    def learner(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ):
        """The Learning part of the agent."""
        
        # If we are running on multiple accelerator devices, this replicates
        # weights and updates across devices.
        replicator = D3PGAgent.get_replicator(self._accelerator)

        with replicator.scope():
            # Create the networks to optimize (online) and target networks.
            online_networks = D3PGNetworks(
                vehicle_policy_network=self._config.vehicle_policy_network,
                vehicle_critic_network=self._config.vehicle_critic_network,
                vehicle_observation_network=self._config.vehicle_observation_network,
                edge_policy_network=self._config.edge_policy_network,
                edge_critic_network=self._config.edge_critic_network,
                edge_observation_network=self._config.edge_observation_network,
            )
            target_networks = copy.deepcopy(online_networks)

            # Initialize the networks.
            online_networks.init(self._config.environment_spec)
            target_networks.init(self._config.environment_spec)

        dataset = self._agent.make_dataset_iterator(replay)
        counter = counting.Counter(counter, 'learner')
        logger = loggers.make_default_logger(
            'learner', time_delta=self._log_every, steps_key='learner_steps')

        return self._agent.make_learner(
            online_networks=online_networks, 
            target_networks=target_networks,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=True,
        )

    def actor(
        self,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        environment: Optional[vehicularNetworkEnv] = None,  #  Create the environment to interact with actor.
    ) -> acme.EnvironmentLoop:
        """The actor process."""

        # Create the behavior policy.        
        networks = D3PGNetworks(
            vehicle_policy_network=self._config.vehicle_policy_network,
            vehicle_critic_network=self._config.vehicle_critic_network,
            vehicle_observation_network=self._config.vehicle_observation_network,
            edge_policy_network=self._config.edge_policy_network,
            edge_critic_network=self._config.edge_critic_network,
            edge_observation_network=self._config.edge_observation_network,
        )
        networks.init(self._config.environment_spec)

        vehicle_policy_network, edge_policy_network = networks.make_policy(
            environment_spec=self._config.environment_spec,
            sigma=self._config.sigma,
        )

        # Create the agent.
        actor = self._agent.make_actor(
            vehicle_policy_network=vehicle_policy_network,
            edge_policy_network=edge_policy_network,
            environment=environment,
            adder=self._agent.make_adder(replay),
            variable_source=variable_source,
        )

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(counter, 'actor')
        logger = loggers.make_default_logger(
            'actor',
            save_data=False,
            time_delta=self._log_every,
            steps_key='actor_steps')

        # Create the loop to connect environment and agent.
        return acme.EnvironmentLoop(environment, actor, counter, logger)

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        logger: Optional[loggers.Logger] = None,
        environment: Optional[vehicularNetworkEnv] = None,  #  Create the environment to interact with evaluator.
    ):
        """The evaluation process."""

        # Create the behavior policy.
        networks = D3PGNetworks(
            vehicle_policy_network=self._config.vehicle_policy_network,
            vehicle_critic_network=self._config.vehicle_critic_network,
            vehicle_observation_network=self._config.vehicle_observation_network,
            edge_policy_network=self._config.edge_policy_network,
            edge_critic_network=self._config.edge_critic_network,
            edge_observation_network=self._config.edge_observation_network,
        )
        networks.init(self._config.environment_spec)
        vehicle_policy_network, edge_policy_network = networks.make_policy(self._config.environment_spec)

        # Create the agent.
        actor = self._agent.make_actor(
            vehicle_policy_network=vehicle_policy_network,
            edge_policy_network=edge_policy_network,
            environment=environment,
            variable_source=variable_source,
        )

        # Create logger and counter.
        counter = counting.Counter(counter, 'evaluator')
        logger = logger or loggers.make_default_logger(
            'evaluator',
            time_delta=self._log_every,
            steps_key='evaluator_steps',
        )

        # Create the run loop and return it.
        return acme.EnvironmentLoop(environment, actor, counter, logger)

    def build(self, name='mad3pg'):
        """Build the distributed agent topology."""
        program = lp.Program(name=name)

        with program.group('replay'):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group('counter'):
            counter = program.add_node(lp.CourierNode(self.counter))

        if self._max_actor_steps:
            with program.group('coordinator'):
                _ = program.add_node(lp.CourierNode(self.coordinator, counter))

        with program.group('learner'):
            learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

        with program.group('evaluator'):
            program.add_node(lp.CourierNode(self.evaluator, learner, counter))

        if not self._num_caches:
            # Use our learner as a single variable source.
            sources = [learner]
        else:
            with program.group('cacher'):
                # Create a set of learner caches.
                sources = []
                for _ in range(self._num_caches):
                    cacher = program.add_node(
                        lp.CacherNode(
                            learner, refresh_interval_ms=2000, stale_after_ms=4000))
                sources.append(cacher)

        with program.group('actor'):
            # Add actors which pull round-robin from our variable sources.
            for actor_id in range(self._num_actors):
                source = sources[actor_id % len(sources)]
                program.add_node(lp.CourierNode(self.actor, replay, source, counter))

        return program