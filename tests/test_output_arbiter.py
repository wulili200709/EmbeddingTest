from __future__ import annotations

import unittest

from devices.output_arbiter import OutputArbiter


class _FakeIo:
    def __init__(self) -> None:
        self.outputs: dict[str, bool] = {}
        self.batches: list[dict[str, bool]] = []

    def set_output(self, name: str, on: bool) -> None:
        self.outputs[str(name)] = bool(on)

    def set_outputs(self, updates) -> None:
        normalized = {str(name): bool(on) for name, on in dict(updates).items()}
        self.batches.append(normalized)
        self.outputs.update(normalized)

    def set_buzzer(self, on: bool) -> None:
        self.outputs["buzzer"] = bool(on)


class OutputArbiterTests(unittest.TestCase):
    def test_result_timer_cannot_silence_latched_line_fault(self) -> None:
        io = _FakeIo()
        arbiter = OutputArbiter(io)  # type: ignore[arg-type]

        arbiter.set_result_buzzer(True)
        arbiter.set_line_output("buzzer", True)
        arbiter.set_result_buzzer(False)

        self.assertTrue(io.outputs["buzzer"])
        arbiter.set_line_output("buzzer", False)
        self.assertFalse(io.outputs["buzzer"])

    def test_motion_stop_is_forwarded_as_one_batch(self) -> None:
        io = _FakeIo()
        arbiter = OutputArbiter(io)  # type: ignore[arg-type]

        arbiter.set_line_outputs({"conveyor_run": False, "waste_removal": False})

        self.assertEqual(
            io.batches,
            [{"conveyor_run": False, "waste_removal": False}],
        )


if __name__ == "__main__":
    unittest.main()
