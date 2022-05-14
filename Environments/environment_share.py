from typing import List


class shared_data(object):
    
    def __init__(self, time_slots_number) -> None:
        self._time_slots_number = time_slots_number
        self._reward_history: List[List[float]] = [[] for _ in range(self._time_slots_number)]

    def append_reward_at_now(self, now: int, reward: float) -> None:
        self._reward_history[now].append(reward)

    def get_reward_history_at_now(self, now: int) -> List[float]:
        return self._reward_history[now]

    def get_min_reward_at_now(self, now: int) -> float:
        return min(self._reward_history[now])

    def get_max_reward_at_now(self, now: int) -> float:
        return max(self._reward_history[now])