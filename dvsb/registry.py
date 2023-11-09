from typing import Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self.classes: dict[str, Type[T]] = {}

    def register(self, cls: Type[T]) -> Type[T]:
        name = getattr(cls, "__name__")
        self.classes[name] = cls
        return cls

    def __getitem__(self, name: str) -> Type[T]:
        return self.classes[name]
