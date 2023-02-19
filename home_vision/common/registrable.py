"""Define Registrable class"""
# This module is largely inspired by the AllenNLP library
# See github.com/allenai/allennlp/blob/master/allennlp/common/registrable.py

from collections import defaultdict
from typing import Callable, List, Type, TypeVar
from future.utils import iteritems

from .exceptions import AlreadyRegisteredError, NotRegisteredError

T = TypeVar("T", bound="Registrable")

class Registrable:
    """
    Any class that inherits from ``Registrable`` gains access to a named
    registry for its subclasses. To register them, just decorate them with the
    classmethod ``@BaseClass.register(name)``.

    After which you can call ``BaseClass.list_available()`` to get the keys
    for the registered subclasses, and ``BaseClass.by_name(name)`` to get the
    corresponding subclass.

    Note that if you use this class to implement a new ``Registrable``
    abstract class, you must ensure that all subclasses of the abstract class
    are loaded when the module is loaded, because the subclasses register
    themselves in their respective files. You can achieve this by having the
    abstract class and all subclasses in the __init__.py of the module in
    which they reside (as this causes any import of either the abstract class
    or a subclass to load all other subclasses and the abstract class).
    """
    _registry = defaultdict(dict)

    @classmethod
    def register(cls: Type[T], name: str, override: bool=False):
        """Decorator used to add the decorated subclass to the registry of the
        base class

        Args:
            name (str): name use to identify the registered subclass
            override (bool, optional): this parameter controls the behavior in
                case where a subclass is registered with the same identifier.
                If True, then the previous subclass will be unregistered in
                profit of the new subclass.

        Raises:
            AlreadyRegisteredError: when ``override`` is False, while trying
                to register a subclass with a name already used by another
                registered subclass
        """
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an error if key has already been used.
            if not override and name in registry:
                raise AlreadyRegisteredError(name, cls, registry[name])
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def registered_name(cls, registered_class: Type[T]) -> str:
        """Get name of the registered class

        Args:
            registered_class (Type[T]): class that needs query name

        Raises:
            NotRegisteredError: when the class is not registered

        Returns:
            str: name of the registered class
        """
        for name, subclass in iteritems(Registrable._registry[cls]):
            if subclass == registered_class:
                return name
        raise NotRegisteredError(cls, registered_cls=registered_class)

    @classmethod
    def by_name(cls, name: str) -> Callable[..., T]:
        """Get the registered class by name

        Args:
            name (str): name of the registered_class

        Raises:
            NotRegisteredError: when the class is not registered

        Returns:
            Callable[..., T]: registered class
        """
        if name not in Registrable._registry[cls]:
            raise NotRegisteredError(cls, name=name)
        return Registrable._registry[cls][name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered sub-classes under a registerable class"""
        return list(Registrable._registry[cls].keys())
