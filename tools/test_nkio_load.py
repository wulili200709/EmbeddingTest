from __future__ import annotations

import argparse
import configparser
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from devices.io_controller import IoController
from devices.io_mapping import IoMapping
from devices.nkio_errors import nkio_error_name
from devices.nkio_raw import NkioRawLib, find_default_nkio_dll_path


def _is_windows_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _quote_windows_arg(value: str) -> str:
    return subprocess.list2cmdline([str(value)])


def _relaunch_as_admin() -> bool:
    if sys.platform != "win32" or _is_windows_admin():
        return False

    executable = sys.executable
    parameters = " ".join(_quote_windows_arg(arg) for arg in [__file__, *sys.argv[1:]])
    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        executable,
        parameters,
        None,
        1,
    )
    if result <= 32:
        raise RuntimeError(f"Failed to request administrator privileges: ShellExecuteW={result}")
    return True


def _load_runtime_options(mapping_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    result: dict[str, str] = {}
    for key in ("nkio_config_path", "nkio_dll_path"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result[key] = text
    return result


def _selected_nkio_config_from_sdk_bin() -> Path | None:
    select_ini = REPO_ROOT / "NKDIOLC_SDK" / "Bin" / "select.ini"
    if not select_ini.exists():
        return None

    parser = configparser.ConfigParser()
    parser.optionxform = str
    try:
        parser.read(select_ini, encoding="utf-8")
    except Exception:
        return None

    if not parser.has_section("SELECTED"):
        return None

    config_path = str(parser.get("SELECTED", "ConfigPath", fallback="") or "").strip()
    if not config_path:
        return None

    relative_path = config_path.lstrip("/\\").replace("/", "\\")
    candidate = REPO_ROOT / "NKDIOLC_SDK" / "Bin" / Path(relative_path)
    if candidate.exists():
        return candidate
    return None


def _find_default_config_path() -> Path | None:
    selected = _selected_nkio_config_from_sdk_bin()
    if selected is not None:
        return selected

    candidates = [
        REPO_ROOT / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "x64" / "Debug" / "NP-6133-16I16O" / "nkio_config.ini",
        REPO_ROOT / "NKDIOLC_SDK" / "Bin" / "NP-6133-16I16O" / "nkio_config.ini",
        REPO_ROOT / "NKDIOLC_SDK" / "ConfigFile" / "NP-6133-16I16O" / "nkio_config.ini",
        REPO_ROOT / "NKDIOLC_SDK" / "ConfigFile" / "J1900" / "NP-6133-16I16O" / "nkio_config.ini",
        REPO_ROOT / "NKDIOLC_SDK" / "Bin" / "NP-61x0-16I16O" / "nkio_config.ini",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_mapping_path(path_text: str | None) -> Path:
    if path_text:
        return Path(path_text).expanduser().resolve()
    return (PROJECT_ROOT / "config" / "defaults" / "io_mapping.json").resolve()


def _resolve_config_path(args, runtime_options: dict[str, str]) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    configured = runtime_options.get("nkio_config_path")
    if configured:
        return Path(configured).expanduser().resolve()
    fallback = _find_default_config_path()
    if fallback is None:
        raise FileNotFoundError("Could not resolve nkio_config.ini")
    return fallback.resolve()


def _resolve_dll_path(args, runtime_options: dict[str, str]) -> Path:
    if args.dll:
        return Path(args.dll).expanduser().resolve()
    configured = runtime_options.get("nkio_dll_path")
    if configured:
        return Path(configured).expanduser().resolve()
    return find_default_nkio_dll_path().resolve()


def _print_environment(mapping_path: Path, config_path: Path, dll_path: Path, mapping: IoMapping, verbose: bool) -> None:
    print("=== NKIO Load Test ===")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"platform: {sys.platform}")
    print(f"is_admin: {_is_windows_admin()}")
    print(f"cwd: {Path.cwd()}")
    print(f"mapping_path: {mapping_path}")
    print(f"mapping_exists: {mapping_path.exists()}")
    print(f"nkio_config_path: {config_path}")
    print(f"nkio_config_exists: {config_path.exists()}")
    print(f"nkio_dll_path: {dll_path}")
    print(f"nkio_dll_exists: {dll_path.exists()}")
    print(f"di_names: {mapping.di_names()}")
    print(f"do_names: {mapping.do_names()}")
    if verbose:
        print("path_preview:")
        for item in os.environ.get("PATH", "").split(os.pathsep)[:10]:
            print(f"  {item}")


def _print_mapping_detail(mapping: IoMapping) -> None:
    print("di_mapping:")
    for name in mapping.di_names():
        cfg = mapping.get_input(name)
        print(f"  {name}: channel={cfg.channel}, active_high={cfg.active_high}")
    print("do_mapping:")
    for name in mapping.do_names():
        cfg = mapping.get_output(name)
        print(f"  {name}: channel={cfg.channel}, active_high={cfg.active_high}")


def _run_raw_probe(config_path: Path, dll_path: Path) -> int:
    print("--- raw_probe ---")
    raw = NkioRawLib(dll_path=dll_path)
    print(f"loaded_dll: {raw.dll_path}")

    init_ret = raw.library_init(config_path)
    print(f"library_init_ret: {init_ret} ({nkio_error_name(init_ret)})")
    if init_ret != 0:
        return init_ret

    try:
        di_ret, di_word = raw.read_di_word(0)
        do_ret, do_word = raw.read_do_word(0)
        print(f"read_di_word_ret: {di_ret} ({nkio_error_name(di_ret)}), value=0x{di_word:04X}")
        print(f"read_do_word_ret: {do_ret} ({nkio_error_name(do_ret)}), value=0x{do_word:04X}")
    finally:
        raw.library_deinit()
        print("library_deinit: done")
    return 0


def _run_controller_probe(
    mapping_path: Path,
    config_path: Path,
    dll_path: Path,
    pulse_output: str | None,
    pulse_seconds: float,
) -> int:
    print("--- controller_probe ---")
    controller = IoController.from_config_file(config_path, mapping_path, dll_path=dll_path)
    controller.open()
    try:
        print(f"snapshot_inputs: {controller.snapshot_inputs()}")
        print(f"snapshot_outputs: {controller.snapshot_outputs()}")
        if pulse_output:
            current = controller.read_output(pulse_output)
            print(f"pulse_output_before: {pulse_output}={current}")
            controller.set_output(pulse_output, True)
            print(f"pulse_output_on: {pulse_output}={controller.read_output(pulse_output)}")
            time.sleep(max(0.0, pulse_seconds))
            controller.set_output(pulse_output, current)
            print(f"pulse_output_restore: {pulse_output}={controller.read_output(pulse_output)}")
    finally:
        controller.close()
        print("controller_close: done")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone NKIO load test for DI/DO troubleshooting.")
    parser.add_argument("--mapping", help="Path to io_mapping.json")
    parser.add_argument("--config", help="Path to nkio_config.ini")
    parser.add_argument("--dll", help="Path to NKIOLIBx64.dll")
    parser.add_argument("--raw-only", action="store_true", help="Only test DLL load and NKDIO_LibraryInit")
    parser.add_argument("--pulse-output", help="Mapped DO name to briefly set on, then restore")
    parser.add_argument("--pulse-seconds", type=float, default=0.5, help="Pulse duration in seconds")
    parser.add_argument("--no-admin-relaunch", action="store_true", help="Do not auto-relaunch as administrator")
    parser.add_argument("--verbose", action="store_true", help="Print additional PATH diagnostics")
    parser.add_argument("--pause", action="store_true", help="Wait for Enter before exiting")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.no_admin_relaunch and _relaunch_as_admin():
        return 0

    mapping_path = _resolve_mapping_path(args.mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing io mapping file: {mapping_path}")

    runtime_options = _load_runtime_options(mapping_path)
    config_path = _resolve_config_path(args, runtime_options)
    dll_path = _resolve_dll_path(args, runtime_options)
    mapping = IoMapping.from_json_file(mapping_path)

    if args.pulse_output and args.pulse_output not in mapping.do_names():
        raise KeyError(f"Unknown DO mapping: {args.pulse_output}")

    _print_environment(mapping_path, config_path, dll_path, mapping, args.verbose)
    _print_mapping_detail(mapping)

    raw_ret = _run_raw_probe(config_path, dll_path)
    if raw_ret != 0:
        return raw_ret
    if args.raw_only:
        return 0

    return _run_controller_probe(
        mapping_path=mapping_path,
        config_path=config_path,
        dll_path=dll_path,
        pulse_output=args.pulse_output,
        pulse_seconds=args.pulse_seconds,
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = int(main())
    except SystemExit as exc:
        code = exc.code
        exit_code = int(code) if isinstance(code, int) else 0
    except Exception:
        traceback.print_exc()
        exit_code = 99
    finally:
        if "--pause" in sys.argv:
            input("Press Enter to exit...")
    raise SystemExit(exit_code)
