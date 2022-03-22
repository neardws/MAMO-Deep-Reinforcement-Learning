from tkinter import E
from dataStruct import applicationList
from dataStruct import viewList
from dataStruct import informationList
import numpy as np

class informationRequirements(object):
    """
    This class is used to store the data requirements of the environment.
    """
    def __init__(
        self,
        max_timestampes: int,
        max_application_number: int,
        min_application_number: int,
        application: applicationList,
        view: viewList,
        information: informationList,
        seed: int
        ) -> None:
        """ initialize the information set.
        Args:
            max_timestampes: the maximum timestamp.
            max_application_number: the maximum application number at each timestampe.
            min_application_number: the minimum application number at each timestampe.
            application: the application list.
            view: the view list.
            information: the information set.
            seed: the random seed.
        """
        self._max_timestampes = max_timestampes
        self._max_application_number = max_application_number
        self._min_application_number = min_application_number
        self._application = application
        self._view = view
        self._information = information
        self._seed = seed

        self.applications_at_time = self.applications_at_times()
        

    def get_max_timestampes(self) -> int:
        """ get the maximum timestamp.
        Returns:
            the maximum timestamp.
        """
        return self._max_timestampes

    def get_application(self) -> applicationList:
        """ get the application list.
        Returns:
            the application list.
        """
        return self._application

    def get_view(self) -> viewList:
        """ get the view list.
        Returns:
            the view list.
        """
        return self._view
    
    def get_information(self) -> informationList:
        """ get the information set.
        Returns:
            the information set.
        """
        return self._information

    def get_seed(self) -> int:
        """ get the random seed.
        Returns:
            the random seed.
        """
        return self._seed
    
    def applications_at_times(self) -> list:
        """ get the applications at each time.
        Returns:
            the applications at times.
        """
        random_application_number = np.random.randint(
            low=self._min_application_number, 
            high=self._max_application_number, 
            size=self._max_timestampes
        )

        applications_at_times = []
        for _ in range(self._max_timestampes):
            applications_at_times.append(
                list(np.random.choice(
                    list(range(self._application.get_number())), 
                    random_application_number[_], 
                    replace=False))
            )

        return applications_at_times
    
    def applications_at_now(self, nowTimeStamp: int) -> list:
        """ get the applications now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the applications list.
        """
        if self.applications_at_time is None:
            return Exception("applications_at_time is None.")
        return self.applications_at_time[nowTimeStamp]

    def views_required_by_application_at_now(self, nowTimeStamp: int) -> list:
        """ get the views required by application now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the views required by application list.
        """
        applications_at_now = self.applications_at_now(nowTimeStamp)
        views_required_by_application_at_now = []
        for _ in applications_at_now:
            views_required_by_application_at_now.append(
                self._application.get_application_list()[_]
            )
        return views_required_by_application_at_now
    
    def information_required_by_views_at_now(self, nowTimeStamp: int) -> list:
        """ get the information required by views now.
        Args:
            nowTimeStamp: the current timestamp.
        Returns:
            the information set required by views list.
        """
        views_required_by_application_at_now = self.views_required_by_application_at_now(nowTimeStamp)
        information_type_required_by_views_at_now = set()

        for _ in views_required_by_application_at_now:
            view_list = self._view.get_view_list()[_]
            for __ in view_list:
                information_type_required_by_views_at_now.add(
                    self._information.get_information_list()[__]["type"]
                )

        information_required_by_views_at_now = []
        for _ in information_type_required_by_views_at_now:
            for __ in self._information.get_information_list():
                if __["type"] == _:
                    information_required_by_views_at_now.append(__)

        return information_required_by_views_at_now

if __name__ == "__main__":
    application = applicationList(
        application_number=10,
        view_number=10,
        views_per_application=1,
        seed=1
    )
    print("application:\n", application.get_application_list())

    information_list = informationList(
        information_number=10, 
        seed=0, 
        data_size_low_bound=0, 
        data_size_up_bound=1, 
        data_types_number=3, 
        update_interval_low_bound=1, 
        update_interval_up_bound=3
    )
    print("information_list:\n", information_list.get_information_list())

    view = viewList(
        view_number=10,
        information_number=10,
        max_information_number=3,
        seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    print("views:\n", view.get_view_list())

    information_required = informationRequirements(
        max_timestampes=10,
        max_application_number=10,
        min_application_number=1,
        application=application,
        view=view,
        information=information_list,
        seed=0
    )
    print(information_required.applications_at_time)
    nowTimeStamp = 0
    print(information_required.applications_at_now(nowTimeStamp))
    print(information_required.views_required_by_application_at_now(nowTimeStamp))
    print(information_required.information_required_by_views_at_now(nowTimeStamp))

    """Print the result."""
    """
    application:
    [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]

    information_list:
    [{'type': 2, 'data_size': 0.5488135039273248, 'update_interval': 1}, 
    {'type': 8, 'data_size': 0.7151893663724195, 'update_interval': 2}, 
    {'type': 4, 'data_size': 0.6027633760716439, 'update_interval': 2}, 
    {'type': 9, 'data_size': 0.5448831829968969, 'update_interval': 1}, 
    {'type': 1, 'data_size': 0.4236547993389047, 'update_interval': 2}, 
    {'type': 6, 'data_size': 0.6458941130666561, 'update_interval': 2}, 
    {'type': 7, 'data_size': 0.4375872112626925, 'update_interval': 2}, 
    {'type': 3, 'data_size': 0.8917730007820798, 'update_interval': 2}, 
    {'type': 0, 'data_size': 0.9636627605010293, 'update_interval': 2}, 
    {'type': 5, 'data_size': 0.3834415188257777, 'update_interval': 2}]

    views:
    [
    [2, 9], 
    [4, 1], 
    [5], 
    [3], 
    [9, 5], 
    [8, 1], 
    [8, 5], 
    [8, 6], 
    [8, 4], 
    [8]]

    [[7], 
    [0, 4], 
    [0, 5, 9, 2, 3, 4, 8, 7, 1], 
    [5], 
    [5, 4, 7, 9, 1, 3, 8, 6, 0], 
    [1, 2, 7, 3, 6, 5, 9], 
    [7, 4, 1, 3, 8], 
    [1, 7, 8, 4], 
    [5], 
    [3, 8, 6, 4, 7]]

    [7]

    [7]

    [{'type': 0, 'data_size': 0.9636627605010293, 'update_interval': 2}, {'type': 7, 'data_size': 0.4375872112626925, 'update_interval': 2}]
    """