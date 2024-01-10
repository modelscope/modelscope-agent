from abc import ABC, abstractmethod


class BaseStorage(ABC):

    @abstractmethod
    def put(self, key: str, value: str):
        """
        save one key value pair
        Args:
            key:
            value:

        Returns:

        """
        pass

    @abstractmethod
    def get(self, key: str, re_load: bool = True):
        """
        get one value by key
        Args:
            key:
            re_load:

        Returns:

        """
        pass

    @abstractmethod
    def delete(self, key):
        """
        delete one key value pair
        Args:
            key:

        Returns:

        """
        pass

    def scan(self):
        """
        get list of object
        Returns:

        """
