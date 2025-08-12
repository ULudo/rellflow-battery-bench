class NoBatteryController(ControllerInterface):
    def reset(self, obs: Any, info: ControllerInfo) -> None:  # noqa: ANN401
        return None

    def act(self, obs: Any, info: ControllerInfo):  # noqa: ANN401
        return 0