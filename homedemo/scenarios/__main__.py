"""`python -m homedemo.scenarios [--dir DIR] [--markdown]`: validate every library file and list it.

Exit status 1 when any scenario is invalid. With --markdown the table is printed in the README format.
"""
from __future__ import annotations

import argparse
import sys

from . import load_home, load_scenarios, summary, validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m homedemo.scenarios")
    parser.add_argument("--dir", default=None, help="scenario folder (default: the bundled library)")
    parser.add_argument("--markdown", action="store_true", help="print the table as Markdown")
    args = parser.parse_args(argv)

    home = load_home()
    scenarios = load_scenarios(args.dir)
    bad = 0
    rows = []
    for sc in scenarios:
        errors = validate(sc, home)
        bad += bool(errors)
        s = summary(sc)
        rows.append((s["id"], s["title"], ",".join(s["tags"]), s["steps"], errors))
    if args.markdown:
        print("| id | title | tags | steps |")
        print("|---|---|---|---|")
        for sid, title, tags, n, _ in rows:
            print(f"| `{sid}` | {title} | {tags} | {n} |")
    else:
        width = max((len(r[0]) for r in rows), default=10)
        for sid, title, tags, n, errors in rows:
            flag = "OK " if not errors else "BAD"
            print(f"{flag} {sid:<{width}} {n:>3} steps  [{tags}]  {title}")
            for err in errors:
                print(f"      - {err}")
    print(f"{len(scenarios)} scenarios, {bad} invalid", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
