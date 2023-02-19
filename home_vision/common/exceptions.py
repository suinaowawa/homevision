"""Custom exceptions"""
class HomeVisionError(Exception):
    """Base class for exceptions raised in the snips-nlu library"""



class AlreadyRegisteredError(HomeVisionError):
    """Raised when attempting to register a subclass which is already
    registered"""

    def __init__(self, name, new_class, existing_class):
        msg = f"Cannot register {name} for {new_class.__name__} as it has already been used to " \
              f"register {existing_class.__name__}"
        super().__init__(msg)


class NotRegisteredError(HomeVisionError):
    """Raised when trying to use a subclass which was not registered"""

    def __init__(self, registrable_cls, name=None, registered_cls=None):
        if name is not None:
            msg = f"'{name}' has not been registered for type {registered_cls}. "
        else:
            msg = f"subclass {registered_cls} has not been registered for type {registrable_cls}. "
        msg += "Use @BaseClass.register('my_component') to register a subclass"
        super().__init__(msg)

class MethodNotExistError(HomeVisionError):
    """Raised when trying to use a method which is not exist"""

    def __init__(self, method_name, available_methods):

        msg = f"'{method_name}' doesn't exist. "

        msg += "Avaiable methods are: " + str(available_methods)
        super().__init__(msg)
