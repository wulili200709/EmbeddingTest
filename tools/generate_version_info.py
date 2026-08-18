from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys


_VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:\.(?P<build>0|[1-9]\d*))?"
    r"(?:[-+][0-9A-Za-z.-]+)?$"
)


def parse_version(value: object) -> tuple[str, tuple[int, int, int, int]]:
    version = str(value or "").strip()
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(
            "version must look like 3.1.0, 3.1.0.4, or 3.1.0-beta.1"
        )
    numbers = tuple(
        int(match.group(name) or 0)
        for name in ("major", "minor", "patch", "build")
    )
    if any(number > 65535 for number in numbers):
        raise ValueError("each numeric Windows version component must be <= 65535")
    return version, numbers


def version_resource_text(
    version: str,
    numeric_version: tuple[int, int, int, int],
    *,
    product_name: str,
    exe_name: str,
) -> str:
    version_tuple = repr(tuple(numeric_version))
    return f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version_tuple},
    prodvers={version_tuple},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0),
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', 'LC System'),
          StringStruct('FileDescription', {product_name!r}),
          StringStruct('FileVersion', {version!r}),
          StringStruct('InternalName', {product_name!r}),
          StringStruct('OriginalFilename', {exe_name!r}),
          StringStruct('ProductName', {product_name!r}),
          StringStruct('ProductVersion', {version!r}),
        ],
      )
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])]),
  ],
)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LC System package version metadata.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--exe-name", required=True)
    parser.add_argument("--version-file", required=True)
    parser.add_argument("--resource-file", required=True)
    parser.add_argument("--manifest-file", required=True)
    args = parser.parse_args()

    try:
        version, numeric_version = parse_version(args.version)
    except ValueError as exc:
        parser.error(str(exc))

    version_path = Path(args.version_file).resolve()
    resource_path = Path(args.resource_file).resolve()
    manifest_path = Path(args.manifest_file).resolve()
    for path in (version_path, resource_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    version_path.write_text(version + "\n", encoding="utf-8")
    resource_path.write_text(
        version_resource_text(
            version,
            numeric_version,
            product_name=str(args.product_name),
            exe_name=str(args.exe_name),
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "product_name": str(args.product_name),
                "version": version,
                "windows_version": list(numeric_version),
                "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated package metadata for {args.product_name} V{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
