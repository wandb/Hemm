from abc import ABC, abstractmethod
from typing import Any, Dict

import weave


class BaseMetric(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, metadata: weave.Model) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate_async(self, metadata: weave.Model) -> Dict[str, Any]:
        pass
