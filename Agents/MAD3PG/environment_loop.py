"""A simple agent-environment training loop."""

import operator
import time
from typing import Optional, Sequence
from Agents.MAD3PG import base
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals
from dm_env import specs
import numpy as np
import tree
from Log.logger import myapp


class EnvironmentLoop(base.Worker):
    """A simple RL environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. Agent is updated if `should_update=True`. This can be used as:

        loop = EnvironmentLoop(environment, actor)
        loop.run(num_episodes)

    A `Counter` instance can optionally be given in order to maintain counts
    between different Acme components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.

    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger. A string `label` can be passed to easily
    change the label associated with the default logger; this is ignored if a
    `Logger` instance is given.

    A list of 'Observer' instances can be specified to generate additional metrics
    to be logged by the logger. They have access to the 'Environment' instance,
    the current timestep datastruct and the current action.
    """

    def __init__(
        self,
        environment,
        actor: base.Actor,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        should_update: bool = True,
        label: str = 'environment_loop',
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._label = label
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self._should_update = should_update
        self._observers = observers

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get an
        observation and then give that observation to the agent in order to retrieve
        an action.

        Returns:
        An instance of `loggers.LoggingData`.
        """
        # Reset any counts and start the environment.
        
        start_time = time.time()
        # myapp.debug("**********************************************************")
        # myapp.debug(f"{self._label} start_time: {start_time}")
        episode_steps = 0

        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode.
        episode_return = tree.map_structure(_generate_zeros_from_spec,
                                            self._environment.reward_spec())
        timestep = self._environment.reset()

        # myapp.debug(f"{self._label} reset time taken: {time.time() - start_time}")
        # start_time = time.time()
        # Make the first observation.
        self._actor.observe_first(timestep)
        for observer in self._observers:
            # Initialize the observer with the current state of the env after reset
            # and the initial timestep.
            observer.observe_first(self._environment, timestep)

        # myapp.debug(f"{self._label} observer time taken: {time.time() - start_time}")
        # start_time = time.time()
        # Run an episode.
        select_action_taken_time = 0
        environment_step_taken_time = 0
        actor_observe_taken_time = 0
        actor_update_taken_time = 0
        transform_action_array_to_actions_taken_times = 0
        compute_baseline_information_objects_taken_times = 0
        compute_baseline_reward_taken_times = 0
        compute_vehicle_information_objects_taken_times = 0
        compute_vehicle_reward_taken_times = 0
        reward_history_taken_times = 0
        update_information_objects_taken_times = 0
        observation_taken_times = 0
        
        baseline_reward_compute_the_timeliness_of_views_times = 0
        baseline_reward_compute_the_consistency_of_views_times = 0
        baseline_reward_compute_the_redundancy_of_views_times = 0
        baseline_reward_compute_the_cost_of_views_times = 0
        baseline_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times = 0
        baseline_reward_compute_the_age_of_view_times = 0
        vehicle_reward_compute_the_timeliness_of_views_times = 0
        vehicle_reward_compute_the_consistency_of_views_times = 0
        vehicle_reward_compute_the_redundancy_of_views_times = 0
        vehicle_reward_compute_the_cost_of_views_times = 0
        vehicle_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times = 0
        vehicle_reward_compute_the_age_of_view_times = 0
        while not timestep.last():
            # Generate an action from the agent's policy and step the environment.
            start_time = time.time()
            action = self._actor.select_action(timestep.observation, timestep.vehicle_observation)
            select_action_taken_time += time.time() - start_time
            # myapp.debug(f"{self._label} select_action time taken: {time.time() - start_time}")
            start_time = time.time()
            timestep, transform_action_array_to_actions_taken_time, \
                compute_baseline_information_objects_taken_time, \
                compute_baseline_reward_taken_time, \
                compute_vehicle_information_objects_taken_time, \
                compute_vehicle_reward_taken_time, \
                reward_history_taken_time, \
                update_information_objects_taken_time, \
                observation_taken_time = self._environment.step(action)
            transform_action_array_to_actions_taken_times += transform_action_array_to_actions_taken_time
            compute_baseline_information_objects_taken_times += compute_baseline_information_objects_taken_time
            compute_baseline_reward_taken_times += compute_baseline_reward_taken_time
            compute_vehicle_information_objects_taken_times += compute_vehicle_information_objects_taken_time
            compute_vehicle_reward_taken_times += compute_vehicle_reward_taken_time
            reward_history_taken_times += reward_history_taken_time
            update_information_objects_taken_times += update_information_objects_taken_time
            observation_taken_times += observation_taken_time
            
            
            # baseline_reward_compute_the_timeliness_of_views_times += baseline_reward_compute_the_timeliness_of_views_time
            # baseline_reward_compute_the_consistency_of_views_times += baseline_reward_compute_the_consistency_of_views_time
            # baseline_reward_compute_the_redundancy_of_views_times += baseline_reward_compute_the_redundancy_of_views_time
            # baseline_reward_compute_the_cost_of_views_times += baseline_reward_compute_the_cost_of_views_time
            # baseline_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times += baseline_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_time
            # baseline_reward_compute_the_age_of_view_times += baseline_reward_compute_the_age_of_view_time
            # vehicle_reward_compute_the_timeliness_of_views_times += vehicle_reward_compute_the_timeliness_of_views_time
            # vehicle_reward_compute_the_consistency_of_views_times += vehicle_reward_compute_the_consistency_of_views_time
            # vehicle_reward_compute_the_redundancy_of_views_times += vehicle_reward_compute_the_redundancy_of_views_time
            # vehicle_reward_compute_the_cost_of_views_times += vehicle_reward_compute_the_cost_of_views_time
            # vehicle_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times += vehicle_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_time
            # vehicle_reward_compute_the_age_of_view_times += vehicle_reward_compute_the_age_of_view_time
            
            environment_step_taken_time += time.time() - start_time
            # myapp.debug(f"{self._label} environment.step time taken: {time.time() - start_time}")
            start_time = time.time()
            # Have the agent observe the timestep and let the actor update itself.
            self._actor.observe(action=action, next_timestep=timestep)
            for observer in self._observers:
                # One environment step was completed. Observe the current state of the
                # environment, the current timestep and the action.
                observer.observe(self._environment, timestep, action)
            actor_observe_taken_time += time.time() - start_time
            # myapp.debug(f"{self._label} actor.observe time taken: {time.time() - start_time}")
            start_time = time.time()
            if self._should_update:
                self._actor.update()
            actor_update_taken_time += time.time() - start_time
            # myapp.debug(f"{self._label} actor.update time taken: {time.time() - start_time}")
            # start_time = time.time()

        myapp.debug(f"{self._label} select_action_taken_time: {select_action_taken_time}")
        
        myapp.debug(f"{self._label} transform_action_array_to_actions_taken_times: {transform_action_array_to_actions_taken_times}")
        myapp.debug(f"{self._label} compute_baseline_information_objects_taken_times: {compute_baseline_information_objects_taken_times}")
        myapp.debug(f"{self._label} compute_baseline_reward_taken_times: {compute_baseline_reward_taken_times}")
        myapp.debug(f"{self._label} compute_vehicle_information_objects_taken_times: {compute_vehicle_information_objects_taken_times}")
        myapp.debug(f"{self._label} compute_vehicle_reward_taken_times: {compute_vehicle_reward_taken_times}")
        myapp.debug(f"{self._label} reward_history_taken_times: {reward_history_taken_times}")
        myapp.debug(f"{self._label} update_information_objects_taken_times: {update_information_objects_taken_times}")
        myapp.debug(f"{self._label} observation_taken_times: {observation_taken_times}")
        
        
        # myapp.debug(f"{self._label} baseline_reward_compute_the_timeliness_of_views_times: {baseline_reward_compute_the_timeliness_of_views_times}")
        # myapp.debug(f"{self._label} baseline_reward_compute_the_consistency_of_views_times: {baseline_reward_compute_the_consistency_of_views_times}")
        # myapp.debug(f"{self._label} baseline_reward_compute_the_redundancy_of_views_times: {baseline_reward_compute_the_redundancy_of_views_times}")
        # myapp.debug(f"{self._label} baseline_reward_compute_the_cost_of_views_times: {baseline_reward_compute_the_cost_of_views_times}")
        # myapp.debug(f"{self._label} baseline_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times: {baseline_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times}")
        # myapp.debug(f"{self._label} baseline_reward_compute_the_age_of_view_times: {baseline_reward_compute_the_age_of_view_times}")
        
        # myapp.debug(f"{self._label} vehicle_reward_compute_the_timeliness_of_views_times: {vehicle_reward_compute_the_timeliness_of_views_times}")
        # myapp.debug(f"{self._label} vehicle_reward_compute_the_consistency_of_views_times: {vehicle_reward_compute_the_consistency_of_views_times}")
        # myapp.debug(f"{self._label} vehicle_reward_compute_the_redundancy_of_views_times: {vehicle_reward_compute_the_redundancy_of_views_times}")
        # myapp.debug(f"{self._label} vehicle_reward_compute_the_cost_of_views_times: {vehicle_reward_compute_the_cost_of_views_times}")
        # myapp.debug(f"{self._label} vehicle_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times: {vehicle_reward_normalize_the_timeliness_consistency_redundancy_and_cost_of_views_times}")
        # myapp.debug(f"{self._label} vehicle_reward_compute_the_age_of_view_times: {vehicle_reward_compute_the_age_of_view_times}")
        
        myapp.debug(f"{self._label} environment_step_taken_time: {environment_step_taken_time}")
        
        
        
        myapp.debug(f"{self._label} actor_observe_taken_time: {actor_observe_taken_time}")
        myapp.debug(f"{self._label} actor_update_taken_time: {actor_update_taken_time}")
        # Book-keeping.
        episode_steps += 1
        # Equivalent to: episode_return += timestep.reward
        # We capture the return value because if timestep.reward is a JAX
        # DeviceArray, episode_return will not be mutated in-place. (In all other
        # cases, the returned episode_return will be the same object as the
        # argument episode_return.)
        episode_return = tree.map_structure(operator.iadd,
                                            episode_return,
                                            timestep.reward)

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        result = {
            'label': self._label,
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
        }
        result.update(counts)
        # myapp.debug(f"{self._label} one episode finished")
        # myapp.debug("**********************************************************")
        for observer in self._observers:
            result.update(observer.get_metrics())
        return result

    def run(self,
            num_episodes: Optional[int] = None,
            num_steps: Optional[int] = None):
        """Perform the run loop.

        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.

        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.

        Args:
        num_episodes: number of episodes to run the loop for.
        num_steps: minimal number of steps to run the loop for.

        Raises:
        ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """
        myapp.debug(f"{self._label} run() num_episodes: {num_episodes}, num_steps: {num_steps}")
        
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return ((num_episodes is not None and episode_count >= num_episodes) or
                    (num_steps is not None and step_count >= num_steps))
    
        episode_count, step_count = 0, 0
        with signals.runtime_terminator():
            while not should_terminate(episode_count, step_count):
                start_time = time.time()
                result = self.run_episode()
                episode_count += 1
                step_count += result['episode_length']
                # Log the given episode results.
                self._logger.write(result)
                myapp.debug(f"{self._label} run() finished time taken: {time.time() - start_time}")
# Placeholder for an EnvironmentLoop alias


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)

