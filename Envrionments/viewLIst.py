import numpy as np

class viewList(object):
    """ the view list. """
    def __init__(
        self, 
        number: int, 
        data_types_number: int, 
        data_types_number_up_bound: int, 
        seeds: list) -> None:
        """ initialize the view set.
        Args:
            number: the number of view set.
            data_types_number: the max number of data types.
            data_types_number_up_bound: the up bound of the data types number.
            seed: the random seed.
        """
        self._number = number
        self._data_types_number = data_types_number
        self._data_types_number_up_bound = data_types_number_up_bound
        self._seeds = seeds

        if self._data_types_number_up_bound > self._data_types_number:
            raise ValueError("The data types number up bound must be less than or equal to the data types number.")

        if len(self._seeds) != self._number:
            raise ValueError("The number of seeds must be equal to the number of view sets.")

        self.view_list = list()

        np.random.seed(self._seeds[0])
        self._random_information_number = np.random.randint(
            size=self._number,
            low=1,
            high=self._data_types_number_up_bound
        )

        for _ in range(self._number):
            information_number = self._random_information_number[_]
            np.random.seed(self._seeds[_])
            self.view_list.append(
                list(np.random.choice(
                    a=self._data_types_number, 
                    size=information_number,
                    replace=False
                ))
            )
        
    def get_view_list(self) -> list:
        """ get the view list.
        Returns:
            the view list.
        """
        return self.view_list

    def set_view_list(self, view_list: list) -> None:
        """ set the view list.
        Args:
            view_list: the view list.
        """
        self.view_list = view_list


if __name__ == "__main__":
    view = viewList(
        number=10,
        data_types_number=10,
        data_types_number_up_bound=10,
        seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    print(view.get_view_list())
    """ Print Example:
    [[2, 9, 6, 4, 0, 3], 
    [4, 1, 5, 0, 7, 2, 3, 6, 9], 
    [5, 4, 1, 2, 9, 6], 
    [3], 
    [9], 
    [8, 1], 
    [8, 5, 0, 2, 1, 9, 7, 3], 
    [8, 6, 9, 0, 2, 5, 7], 
    [8, 4, 7], 
    [8, 2, 5, 6, 3]]
    """

