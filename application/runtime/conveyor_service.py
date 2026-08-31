from __future__ import annotations

from concurrent.futures import Future

from . import conveyor


class RuntimeConveyorService:
    """Object-composed facade for conveyor runtime operations."""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def load_config(self):
        return conveyor._load_conveyor_config(self._runtime)

    def initialize(self, input_snapshot=None) -> bool:
        return conveyor._initialize_conveyor_controller(self._runtime, input_snapshot)

    def shutdown(self) -> None:
        conveyor._shutdown_conveyor_controller(self._runtime)

    def wait_for_inspections(self) -> None:
        conveyor._wait_for_conveyor_inspections(self._runtime)

    def publish_state(self, snapshot) -> None:
        conveyor._publish_conveyor_state(self._runtime, snapshot)

    def tick(self) -> None:
        conveyor._tick_conveyor(self._runtime)

    def handle_di_event(self, name: str, state: bool) -> None:
        conveyor._handle_conveyor_di_event(self._runtime, name, state)

    def handle_io_error(self, name: str, detail: str) -> None:
        conveyor._handle_conveyor_io_error(self._runtime, name, detail)

    def enqueue_inspection(self, sequence_id: int, epoch: int) -> None:
        conveyor._enqueue_conveyor_inspection(self._runtime, sequence_id, epoch)

    def prepare_start(self) -> tuple[bool, str]:
        return conveyor._prepare_conveyor_start(self._runtime)

    def run_capture(self, sequence_id: int, epoch: int) -> list[dict[str, object]]:
        return conveyor._run_conveyor_capture(self._runtime, sequence_id, epoch)

    def capture_task_finished(self, sequence_id: int, epoch: int, done: Future) -> None:
        conveyor._on_conveyor_capture_task_finished(self._runtime, sequence_id, epoch, done)

    def run_inspection(
        self,
        sequence_id: int,
        epoch: int,
        captured: list[dict[str, object]],
    ) -> tuple[str, str]:
        return conveyor._run_conveyor_inspection(
            self._runtime,
            sequence_id,
            epoch,
            captured,
        )

    def inspection_task_finished(self, sequence_id: int, epoch: int, done: Future) -> None:
        conveyor._on_conveyor_inspection_task_finished(self._runtime, sequence_id, epoch, done)

    def start(self):
        return conveyor.start_conveyor(self._runtime)

    def stop(self):
        return conveyor.stop_conveyor(self._runtime)

    def start_purge(self):
        return conveyor.start_conveyor_purge(self._runtime)

    def continue_purge(self):
        return conveyor.continue_conveyor_purge(self._runtime)

    def acknowledge_alarm(self):
        return conveyor.acknowledge_conveyor_alarm(self._runtime)


__all__ = ["RuntimeConveyorService"]
