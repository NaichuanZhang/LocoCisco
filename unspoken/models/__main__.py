"""python -m unspoken.models show|check [--profile P] [--path FILE] [--roles a,b] [--json] [--timeout S]

  show   the resolved role table for the profile (provider, target, model, label, options) and which
         env / override values were applied. No network.
  check  healthy() + one tiny probe per enabled role, with latency. Exit 1 if any enabled role fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict

from ..contracts import ROLES, RemoteNotAllowed, RoleHealth
from . import ConfigError, load_registry
from .probes import PROBE_TIMEOUT_S, target_of


def _roles(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    roles = [r.strip() for r in arg.split(",") if r.strip()]
    bad = [r for r in roles if r not in ROLES]
    if bad:
        raise SystemExit(f"unknown role(s) {bad}; roles are {list(ROLES)}")
    return roles


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*header)] + [fmt.format(*[str(c) for c in r]) for r in rows])


def cmd_show(reg, roles: list[str] | None, as_json: bool) -> int:
    wanted = roles or list(ROLES)
    if as_json:
        out = {"profile": reg.profile, "local_only": reg.local_only, "path": reg.path, "applied": reg.applied,
               "roles": {r: asdict(reg.spec(r)) for r in wanted}}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    print(f"profile {reg.profile}  local_only={'yes' if reg.local_only else 'no'}  file {reg.path}")
    rows = []
    for r in wanted:
        s = reg.spec(r)
        opts = ", ".join(f"{k}={v}" for k, v in s.options.items()) if s.options else ""
        rows.append([r, s.provider, target_of(s) if s.provider != "none" else "-", s.model or "", reg.label(r), opts])
    print(_table(rows, ["ROLE", "PROVIDER", "TARGET", "MODEL", "LABEL", "OPTIONS"]))
    if reg.applied:
        print("\noverrides applied:")
        for line in reg.applied:
            print(f"  {line}")
    return 0


def _fmt_health(h: RoleHealth) -> list[str]:
    lat = f"{h.latency_ms:.0f} ms" if h.latency_ms is not None else "-"
    return [h.role, h.provider, h.target, "ok" if h.ok else "FAIL", lat, h.detail]


async def cmd_check(reg, roles: list[str] | None, as_json: bool, timeout: float) -> int:
    try:
        health = await reg.check(roles, timeout=timeout)
    finally:
        await reg.aclose()
    failed = [h for h in health if not h.ok]
    if as_json:
        print(json.dumps({"profile": reg.profile, "local_only": reg.local_only, "ok": not failed,
                          "roles": [asdict(h) for h in health]}, indent=2, ensure_ascii=False))
    else:
        print(f"profile {reg.profile}  local_only={'yes' if reg.local_only else 'no'}  file {reg.path}")
        print(_table([_fmt_health(h) for h in health], ["ROLE", "PROVIDER", "TARGET", "OK", "LATENCY", "DETAIL"]))
        enabled = [h for h in health if h.detail != "disabled"]
        print(f"\n{len(enabled) - len(failed)}/{len(enabled)} enabled roles ok"
              + (f"; FAILED: {', '.join(h.role for h in failed)}" if failed else ""))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m unspoken.models", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["show", "check"])
    ap.add_argument("--profile", "-p", default=None, help="profile name (default: $UNSPOKEN_PROFILE or dev)")
    ap.add_argument("--path", default=None, help="models TOML (default: $UNSPOKEN_MODELS or config/models.toml)")
    ap.add_argument("--roles", default=None, help="comma-separated subset of roles")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--timeout", type=float, default=PROBE_TIMEOUT_S, help="per-role probe timeout in seconds")
    a = ap.parse_args(argv)
    try:
        reg = load_registry(a.profile, a.path)
    except (ConfigError, RemoteNotAllowed) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    roles = _roles(a.roles)
    if a.command == "show":
        return cmd_show(reg, roles, a.json)
    return asyncio.run(cmd_check(reg, roles, a.json, a.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
