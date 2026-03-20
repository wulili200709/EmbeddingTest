from __future__ import annotations

import argparse
import time
from pathlib import Path

from devices import IoController, IoMapping


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_default_config_path() -> Path:
    root = repo_root()
    candidates = [
        root / "NKDIOLC_SDK" / "ConfigFile" / "J1900" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "ConfigFile" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "Bin" / "NP-61x0-16I16O" / "nkio_config.ini",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not locate nkio_config.ini. Searched:\n{searched}")


def format_word(value: int) -> str:
    return f"0x{value:04X} ({value:016b})"


def build_mapping(args: argparse.Namespace) -> IoMapping:
    return IoMapping.from_dict(
        {
            "di": {
                "foot_switch": {"channel": int(args.foot_di), "active_high": True},
            },
            "do": {
                "tower_red": {"channel": int(args.red_do), "active_high": True},
                "tower_green": {"channel": int(args.green_do), "active_high": True},
                "tower_blue": {"channel": int(args.blue_do), "active_high": True},
                "light_cam1": {"channel": int(args.light1_do), "active_high": bool(args.light_active_high)},
                "light_cam2": {"channel": int(args.light2_do), "active_high": bool(args.light_active_high)},
            },
        }
    )


def pulse_output(io: IoController, name: str, duration_s: float, label: str) -> None:
    cfg = io.mapping.get_output(name)
    print(f"[TEST] {label}: {name} -> ON (DO{cfg.channel}, active_high={cfg.active_high})")
    io.set_output(name, True)
    time.sleep(duration_s)
    print(f"[TEST] {label}: {name} -> OFF")
    io.set_output(name, False)


def watch_foot_switch(io: IoController, watch_seconds: float, poll_interval_s: float) -> None:
    foot_cfg = io.mapping.get_input("foot_switch")
    print(f"[WATCH] Monitoring foot_switch for {watch_seconds:.1f}s (DI{foot_cfg.channel})")
    prev_state = io.read_foot_switch()
    print(f"[WATCH] Initial state: {'HIGH' if prev_state else 'LOW'}")
    t_end = time.perf_counter() + watch_seconds
    while time.perf_counter() < t_end:
        state = io.read_foot_switch()
        if state != prev_state:
            edge = "RISING" if state and not prev_state else "FALLING"
            print(f"[WATCH] foot_switch {edge}")
            prev_state = state
        time.sleep(poll_interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal NKIO board validation script.")
    parser.add_argument("--config", type=Path, default=None, help="Path to nkio_config.ini")
    parser.add_argument("--foot-di", type=int, default=0, help="Foot switch DI channel")
    parser.add_argument("--red-do", type=int, default=0, help="Tower red DO channel")
    parser.add_argument("--green-do", type=int, default=1, help="Tower green DO channel")
    parser.add_argument("--blue-do", type=int, default=2, help="Tower blue DO channel")
    parser.add_argument("--light1-do", type=int, default=3, help="Camera1 light DO channel")
    parser.add_argument("--light2-do", type=int, default=4, help="Camera2 light DO channel")
    parser.add_argument(
        "--light-active-high",
        action="store_true",
        help="Treat camera lights as active-high. Default assumes active-low until site confirmation.",
    )
    parser.add_argument("--pulse-ms", type=int, default=500, help="Output pulse duration in milliseconds")
    parser.add_argument("--watch-seconds", type=float, default=10.0, help="Foot switch watch time in seconds")
    parser.add_argument("--poll-ms", type=int, default=50, help="Foot switch polling interval in milliseconds")
    parser.add_argument("--skip-output-test", action="store_true", help="Skip DO pulse tests")
    parser.add_argument("--skip-watch-di", action="store_true", help="Skip foot switch monitoring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config is not None else find_default_config_path()
    pulse_s = max(0.01, float(args.pulse_ms) / 1000.0)
    poll_s = max(0.005, float(args.poll_ms) / 1000.0)

    print("[INFO] NKIO minimal validation")
    print(f"[INFO] config = {config_path}")
    print(f"[INFO] foot switch = DI{args.foot_di}")
    print(f"[INFO] tower red/green/blue = DO{args.red_do}/DO{args.green_do}/DO{args.blue_do}")
    print(
        f"[INFO] light cam1/cam2 = DO{args.light1_do}/DO{args.light2_do} "
        f"(active_high={bool(args.light_active_high)})"
    )

    mapping = build_mapping(args)
    io = IoController.default_debug_controller(config_path)
    io.mapping = mapping

    try:
        io.open()
        print("[OK] Board opened")

        di_word = io.board.read_di_word()
        do_word = io.board.read_do_word()
        print(f"[INFO] DI word = {format_word(di_word)}")
        print(f"[INFO] DO word = {format_word(do_word)}")
        print(f"[INFO] Foot DI current state = {'HIGH' if io.read_foot_switch() else 'LOW'}")

        if not args.skip_output_test:
            print("[STEP] Testing tower lights")
            pulse_output(io, "tower_red", pulse_s, "Tower RED")
            pulse_output(io, "tower_green", pulse_s, "Tower GREEN")
            pulse_output(io, "tower_blue", pulse_s, "Tower BLUE")

            print("[STEP] Testing camera lights")
            pulse_output(io, "light_cam1", pulse_s, "Light CAM1")
            pulse_output(io, "light_cam2", pulse_s, "Light CAM2")

        if not args.skip_watch_di:
            watch_foot_switch(
                io,
                watch_seconds=float(args.watch_seconds),
                poll_interval_s=poll_s,
            )

    finally:
        try:
            if io.is_open:
                io.clear_outputs(["tower_red", "tower_green", "tower_blue", "light_cam1", "light_cam2"])
                print("[SAFE] Cleared tested DO outputs")
        finally:
            io.close()
            print("[OK] Board closed")


if __name__ == "__main__":
    main()
