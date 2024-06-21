from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMetric(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        pass
