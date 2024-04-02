from abc import ABC, abstractmethod


class BaseStorage(ABC):

    @abstractmethod
    def add(self, *args, **kwargs):
        """add items to db or indexer"""
        pass

    @abstractmethod
    def search(self, *args, **kwargs):
        """search from db or indexer"""
        pass

    @abstractmethod
    def delete(self, *args, **kwargs):
        """delete data from db or indexer"""
        pass
