"""Define configurable class"""
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

from home_vision.common.registrable import Registrable

class Configurable(with_metaclass(ABCMeta, Registrable)):
    """Base class for homevision modules"""
    @classmethod
    @abstractmethod
    def from_config(cls, config):
        """Abstract method to load class from config"""
        pass
