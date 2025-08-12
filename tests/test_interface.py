from rellflow_battery_bench.control import ControllerInterface, ControllerInfo


def test_controller_protocol_methods():
    class Dummy:
        def reset(self, obs, info: ControllerInfo):
            return None

        def act(self, obs, info: ControllerInfo):
            return 0

    ctrl: ControllerInterface = Dummy()
    assert hasattr(ctrl, "reset")
    assert hasattr(ctrl, "act")
