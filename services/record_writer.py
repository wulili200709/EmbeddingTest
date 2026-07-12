from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TestRecord:
    record_time: str
    product_name: str
    recipe_name: str = ""
    serial_no: str = ""
    final_result: str = ""
    camera1_result: str = ""
    camera2_result: str = ""
    camera3_result: str = ""
    duration_ms: int = 0
    is_error: bool = False
    error_code: str = ""
    error_message: str = ""
    lock_required: bool = False
    release_required: bool = False
    release_result: str = ""
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        *,
        product_name: str,
        recipe_name: str = "",
        serial_no: str = "",
        final_result: str = "",
        camera1_result: str = "",
        camera2_result: str = "",
        camera3_result: str = "",
        duration_ms: int = 0,
        is_error: bool = False,
        error_code: str = "",
        error_message: str = "",
        lock_required: bool = False,
        release_required: bool = False,
        release_result: str = "",
        extra_fields: dict[str, Any] | None = None,
    ) -> "TestRecord":
        return cls(
            record_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            product_name=product_name,
            recipe_name=recipe_name,
            serial_no=serial_no,
            final_result=final_result,
            camera1_result=camera1_result,
            camera2_result=camera2_result,
            camera3_result=camera3_result,
            duration_ms=int(duration_ms),
            is_error=bool(is_error),
            error_code=error_code,
            error_message=error_message,
            lock_required=bool(lock_required),
            release_required=bool(release_required),
            release_result=release_result,
            extra_fields=dict(extra_fields or {}),
        )


@dataclass
class ReleaseLogRecord:
    record_time: str
    product_name: str
    recipe_name: str = ""
    event_type: str = ""
    result: str = ""
    message: str = ""
    runtime_state: str = ""
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        *,
        product_name: str,
        recipe_name: str = "",
        event_type: str = "",
        result: str = "",
        message: str = "",
        runtime_state: str = "",
        extra_fields: dict[str, Any] | None = None,
    ) -> "ReleaseLogRecord":
        return cls(
            record_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            product_name=product_name,
            recipe_name=recipe_name,
            event_type=event_type,
            result=result,
            message=message,
            runtime_state=runtime_state,
            extra_fields=dict(extra_fields or {}),
        )


class CsvRecordWriter:
    """Append one test record per product into a daily CSV file."""

    DEFAULT_COLUMNS = [
        "record_time",
        "product_name",
    ]

    def __init__(self, base_directory: str | Path) -> None:
        self.base_directory = Path(base_directory)

    def file_path_for_date(self, dt: datetime | None = None) -> Path:
        target_dt = dt or datetime.now()
        return self.base_directory / f"{target_dt.strftime('%Y-%m-%d')}.csv"

    def append_record(self, record: TestRecord) -> Path:
        self.base_directory.mkdir(parents=True, exist_ok=True)
        file_path = self.file_path_for_date(datetime.strptime(record.record_time, "%Y-%m-%d %H:%M:%S"))
        row = self._record_to_row(record)
        fieldnames = self._fieldnames_for_row(row)
        file_exists = file_path.exists()

        if file_exists:
            existing_fieldnames = self._existing_fieldnames(file_path)
            if existing_fieldnames:
                fieldnames = self._merge_fieldnames(existing_fieldnames, row)
            if existing_fieldnames and existing_fieldnames != fieldnames:
                existing_rows = self._read_existing_rows(file_path)
                self._rewrite_file(file_path, fieldnames, existing_rows)
                file_exists = True

        with file_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return file_path

    def _record_to_row(self, record: TestRecord) -> dict[str, Any]:
        row = {
            "record_time": record.record_time,
            "product_name": record.product_name,
        }
        row.update(dict(record.extra_fields or {}))
        return row

    def _fieldnames_for_row(self, row: dict[str, Any]) -> list[str]:
        extra_keys = [key for key in row.keys() if key not in self.DEFAULT_COLUMNS]
        return [*self.DEFAULT_COLUMNS, *extra_keys]

    def _merge_fieldnames(
        self,
        existing_fieldnames: list[str],
        row: dict[str, Any],
    ) -> list[str]:
        fieldnames = list(self.DEFAULT_COLUMNS)
        candidate_keys = [*existing_fieldnames, *row.keys()]
        roi_keys = [key for key in candidate_keys if self._is_roi_result_column(key)]
        other_keys = [key for key in candidate_keys if not self._is_roi_result_column(key)]
        for key in [*roi_keys, *other_keys]:
            if key not in fieldnames:
                fieldnames.append(key)
        return fieldnames

    @staticmethod
    def _is_roi_result_column(key: object) -> bool:
        camera_id, separator, roi_name = str(key or "").partition(".")
        return bool(
            separator
            and camera_id.startswith("cam")
            and camera_id[3:].isdigit()
            and roi_name
        )

    @staticmethod
    def _existing_fieldnames(file_path: Path) -> list[str]:
        try:
            with file_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.reader(csv_file)
                return next(reader, [])
        except Exception:
            return []

    @staticmethod
    def _read_existing_rows(file_path: Path) -> list[dict[str, Any]]:
        try:
            with file_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
                return list(csv.DictReader(csv_file))
        except Exception:
            return []

    @staticmethod
    def _rewrite_file(file_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        with file_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


class CsvReleaseLogWriter:
    """Append one release/trigger event per line into a daily CSV file."""

    DEFAULT_COLUMNS = [
        "record_time",
        "product_name",
        "recipe_name",
        "event_type",
        "result",
        "message",
        "runtime_state",
    ]

    def __init__(self, base_directory: str | Path) -> None:
        self.base_directory = Path(base_directory)

    def file_path_for_date(self, dt: datetime | None = None) -> Path:
        target_dt = dt or datetime.now()
        return self.base_directory / f"{target_dt.strftime('%Y-%m-%d')}.csv"

    def append_record(self, record: ReleaseLogRecord) -> Path:
        self.base_directory.mkdir(parents=True, exist_ok=True)
        file_path = self.file_path_for_date(datetime.strptime(record.record_time, "%Y-%m-%d %H:%M:%S"))
        row = self._record_to_row(record)
        fieldnames = self._fieldnames_for_row(row)
        file_exists = file_path.exists()

        with file_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return file_path

    def _record_to_row(self, record: ReleaseLogRecord) -> dict[str, Any]:
        base = asdict(record)
        extra_fields = base.pop("extra_fields", {})
        base.update(extra_fields)
        return base

    def _fieldnames_for_row(self, row: dict[str, Any]) -> list[str]:
        extra_keys = [key for key in row.keys() if key not in self.DEFAULT_COLUMNS]
        return [*self.DEFAULT_COLUMNS, *sorted(extra_keys)]


class TestRecordService:
    def __init__(self, writer: CsvRecordWriter) -> None:
        self.writer = writer

    def write_product_result(
        self,
        *,
        product_name: str,
        final_result: str,
        recipe_name: str = "",
        serial_no: str = "",
        camera1_result: str = "",
        camera2_result: str = "",
        camera3_result: str = "",
        duration_ms: int = 0,
        is_error: bool = False,
        error_code: str = "",
        error_message: str = "",
        lock_required: bool = False,
        release_required: bool = False,
        release_result: str = "",
        extra_fields: dict[str, Any] | None = None,
    ) -> Path:
        record = TestRecord.now(
            product_name=product_name,
            recipe_name=recipe_name,
            serial_no=serial_no,
            final_result=final_result,
            camera1_result=camera1_result,
            camera2_result=camera2_result,
            camera3_result=camera3_result,
            duration_ms=duration_ms,
            is_error=is_error,
            error_code=error_code,
            error_message=error_message,
            lock_required=lock_required,
            release_required=release_required,
            release_result=release_result,
            extra_fields=extra_fields,
        )
        return self.writer.append_record(record)


class ReleaseLogService:
    def __init__(self, writer: CsvReleaseLogWriter) -> None:
        self.writer = writer

    def write_event(
        self,
        *,
        product_name: str,
        recipe_name: str = "",
        event_type: str,
        result: str,
        message: str = "",
        runtime_state: str = "",
        extra_fields: dict[str, Any] | None = None,
    ) -> Path:
        record = ReleaseLogRecord.now(
            product_name=product_name,
            recipe_name=recipe_name,
            event_type=event_type,
            result=result,
            message=message,
            runtime_state=runtime_state,
            extra_fields=extra_fields,
        )
        return self.writer.append_record(record)
