from acme.agents import agent


class RandomAgent(agent.Agent):
    def __init__(self):

        actor = self.make_actor(vehicle_policy_network, edge_policy_network, self._config.environment, adder)
        learner = self.make_learner(online_networks, target_networks, dataset, self._config.counter, self._config.logger, self._config.checkpoint)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(self._config.batch_size, self._config.min_replay_size),
            observations_per_step=float(self._config.batch_size) / self._config.samples_per_insert)


