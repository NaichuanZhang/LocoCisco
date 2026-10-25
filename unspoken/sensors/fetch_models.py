"""Download the sensor models that are not vendored (sha256-pinned).

    python -m unspoken.sensors.fetch_models            # the default speaker model (CAM++)
    python -m unspoken.sensors.fetch_models --all      # also ResNet34-LM and fresh copies of the vendored files
    UNSPOKEN_MODELS_DIR=/data/unspoken/models python -m unspoken.sensors.fetch_models
"""
from __future__ import annotations

import argparse
import sys

from .assets import PINNED, fetch, models_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"files to fetch (default: wespeaker CAM++); known: {', '.join(PINNED)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    names = list(PINNED) if args.all else (args.names or ["wespeaker_en_voxceleb_CAM++.onnx"])
    rc = 0
    for name in names:
        if name not in PINNED:
            print(f"unknown model {name}; known: {', '.join(PINNED)}", file=sys.stderr)
            rc = 2
            continue
        try:
            p = fetch(name, force=args.force)
            print(f"ok  {p} ({p.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"FAIL {name}: {exc}", file=sys.stderr)
            rc = 1
    print(f"models dir: {models_dir()}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
