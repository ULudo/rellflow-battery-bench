from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class ControllerInfo:
    config: Dict[str, Any]
    metadata: Dict[str, Any]


class ControllerInterface(Protocol):
    def reset(self, obs: Any, info: ControllerInfo) -> None:  # noqa: ANN401
        ...

    def act(self, obs: Any, info: ControllerInfo) -> Any:  # noqa: ANN401
        ...
