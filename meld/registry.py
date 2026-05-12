from collections.abc import Callable, MutableMapping
from typing import Any, ClassVar


class Registry[T]:
    """
    Generic registry for storing and retrieving registered elements by key.

    **Type Parameters:**

    T
        Type of registered objects
    """

    # Note: Any value type due to a limitation of the Python type system with ClassVar
    # See: https://typing.python.org/en/latest/spec/class-compat.html#classvar
    _elements: ClassVar[MutableMapping[str, Any]]

    def __init_subclass__(cls):
        cls._elements = {}

    @classmethod
    def register(cls, key: str) -> Callable[[T], T]:
        """
        Creates a decorator to register a target function or class under the given key.

        Args:
            key: The key to register the element under.

        Returns:
            A decorator function that registers the target.

        Raises:
            ValueError: In the returned decorator if the key is already
                registered.
        """

        def registration_decorator(target: T) -> T:
            if key in cls._elements:
                raise ValueError(f"Key {key!r} was already registered")

            cls._elements[key] = target
            return target

        return registration_decorator

    @classmethod
    def get(cls, key: str) -> T:
        """
        Retrieves a registered element by key.

        Args:
            key: The key of the element to retrieve.

        Returns:
            The registered element.
        """

        return cls._elements[key]
