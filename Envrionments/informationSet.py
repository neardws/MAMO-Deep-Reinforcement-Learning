import numpy as np

class informationSet(object):
    """
    This class is used to store the information set of the environment.
    """
    def __init__(
        self, 
        number: int, 
        seed: int, 
        data_size_low_bound: float,
        data_size_up_bound: float,
        data_types_number: int,
        update_interval_low_bound: int,
        update_interval_up_bound: int
        ) -> None:
        """ initialize the information set.
        Args:
            number: the number of information set.
            seed: the random seed.
            data_size_low_bound: the low bound of the data size.
            data_size_up_bound: the up bound of the data size.
            data_types_number: the number of data types.
            update_interval_low_bound: the low bound of the update interval.
            update_interval_up_bound: the up bound of the update interval.
        """
        self._number = number
        self._seed = seed
        self._data_size_low_bound = data_size_low_bound
        self._data_size_up_bound = data_size_up_bound
        self._data_types_number = data_types_number
        self._update_interval_low_bound = update_interval_low_bound
        self._update_interval_up_bound = update_interval_up_bound

        
        if self._data_types_number != self._number:
            self._data_types_number = self._number
        np.random.seed(self._seed)
        self.types_of_information = np.random.permutation(list(range(self._data_types_number)))

        np.random.seed(self._seed)
        self.data_size_of_information = np.random.uniform(
            low=self._data_size_low_bound,
            high=self._data_size_up_bound,
            size=self._number
        )

        np.random.seed(self._seed)
        self.update_interval_of_information = np.random.randint(
            size=self._number, 
            low=self._update_interval_low_bound, 
            high=self._update_interval_up_bound
        )

        self.information_set = set()
        for i in range(self._number):
            self.information_set.add(
                (
                    self.types_of_information[i],
                    self.data_size_of_information[i],
                    self.update_interval_of_information[i]
                )
            )
        
    def get_information_set(self) -> set:
        return self.information_set
    
    def set_information_set(self, information_set) -> None:
        self.information_set = information_set


if __name__ == "__main__":
    information_set = informationSet(
        number=10, 
        seed=0, 
        data_size_low_bound=0, 
        data_size_up_bound=1, 
        data_types_number=3, 
        update_interval_low_bound=1, 
        update_interval_up_bound=3
    )
    print(information_set.get_information_set())