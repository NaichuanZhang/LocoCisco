"""Reads config/models.toml into one RoleSpec per role for the active profile.

Resolution order (first wins):
  profile : load_registry(profile=...) > UNSPOKEN_PROFILE > "dev"
  path    : load_registry(path=...)    > UNSPOKEN_MODELS  > <repo>/config/models.toml
  per key : overrides dict ("llm.url") > env UNSPOKEN_<ROLE>_<KEY> > the profile's TOML block
            > the profile it `extends` > provider "none" (role disabled)

TOML shape (every profile is a table under [profiles]):

    [profiles.gb10]
    local_only = true                 # refuse any provider that is not on this box or its LAN
    extends = "dev"                   # optional: start from another profile's role blocks

    [profiles.gb10.llm]               # one block per role (unspoken.contracts.ROLES)
    provider = "openai"               # see unspoken/models/providers.py
    url = "http://127.0.0.1:8000/v1"
    model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
    label = "Qwen3.6 35B-A3B on vLLM"
    options = { max_tokens = 300 }    # provider-specific keys, passed through as RoleSpec.options

Unknown roles, unknown keys in a role block, and malformed overrides raise ConfigError with the file,
profile and key named, so a typo fails loudly instead of silently disabling a model.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import ROLES, RoleSpec

DEFAULT_PROFILE = "dev"
ENV_PROFILE = "UNSPOKEN_PROFILE"
ENV_PATH = "UNSPOKEN_MODELS"
ENV_PREFIX = "UNSPOKEN_"
ROLE_KEYS: tuple[str, ...] = ("provider", "url", "model", "label")   # overridable scalar keys
ROLE_BLOCK_KEYS: frozenset[str] = frozenset(ROLE_KEYS) | {"options"}
PROFILE_KEYS: frozenset[str] = frozenset({"local_only", "extends"})
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = REPO_ROOT / "config" / "models.toml"


class ConfigError(ValueError):
    """config/models.toml (or an override) is wrong; the message says where."""


@dataclass
class ResolvedProfile:
    profile: str
    path: str
    local_only: bool
    specs: dict[str, RoleSpec]                       # every role in ROLES; disabled ones have provider "none"
    applied: list[str] = field(default_factory=list)  # human lines: which overrides were applied, in order


# --------------------------------------------------------------------------- file
def resolve_profile_name(profile: str | None, environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return profile or env.get(ENV_PROFILE) or DEFAULT_PROFILE


def resolve_path(path: str | None, environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(path or env.get(ENV_PATH) or DEFAULT_PATH)


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"models config not found: {path} (set {ENV_PATH} or pass path=)") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc


def profile_names(doc: dict[str, Any]) -> list[str]:
    profiles = doc.get("profiles")
    if not isinstance(profiles, dict):
        raise ConfigError("models config has no [profiles.*] tables")
    return list(profiles)


def _role_block(path: Path, profile: str, role: str, block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ConfigError(f"{path}: [profiles.{profile}.{role}] must be a table, got {type(block).__name__}")
    bad = sorted(set(block) - ROLE_BLOCK_KEYS)
    if bad:
        raise ConfigError(f"{path}: [profiles.{profile}.{role}] has unknown key(s) {bad}; "
                          f"allowed: {sorted(ROLE_BLOCK_KEYS)} (provider-specific keys go under options)")
    for key in ROLE_KEYS:
        if key in block and not isinstance(block[key], str):
            raise ConfigError(f"{path}: [profiles.{profile}.{role}].{key} must be a string")
    if "options" in block and not isinstance(block["options"], dict):
        raise ConfigError(f"{path}: [profiles.{profile}.{role}].options must be a table")
    if not block.get("provider"):
        raise ConfigError(f"{path}: [profiles.{profile}.{role}] needs a provider (use \"none\" to disable the role)")
    return dict(block)


def _profile_blocks(doc: dict[str, Any], path: Path, profile: str, seen: tuple[str, ...] = ()
                    ) -> tuple[bool, dict[str, dict[str, Any]]]:
    """(local_only, {role: block}) for a profile, with `extends` applied (child blocks replace whole role blocks)."""
    profile_names(doc)                                   # validates the [profiles] table exists
    profiles = doc["profiles"]
    if profile not in profiles:
        raise ConfigError(f"{path}: no profile {profile!r}; have {profile_names(doc)}")
    if profile in seen:
        raise ConfigError(f"{path}: profile {profile!r} extends itself ({' -> '.join(seen + (profile,))})")
    table = profiles[profile]
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: [profiles.{profile}] must be a table")
    unknown = sorted(k for k in table if k not in PROFILE_KEYS and k not in ROLES)
    if unknown:
        raise ConfigError(f"{path}: [profiles.{profile}] has unknown role(s)/key(s) {unknown}; "
                          f"roles are {list(ROLES)}, profile keys are {sorted(PROFILE_KEYS)}")
    local_only = False
    blocks: dict[str, dict[str, Any]] = {}
    parent = table.get("extends")
    if parent is not None:
        if not isinstance(parent, str):
            raise ConfigError(f"{path}: [profiles.{profile}].extends must be a profile name")
        local_only, blocks = _profile_blocks(doc, path, parent, seen + (profile,))
    if "local_only" in table:
        if not isinstance(table["local_only"], bool):
            raise ConfigError(f"{path}: [profiles.{profile}].local_only must be true or false")
        local_only = table["local_only"]
    for role in ROLES:
        if role in table:
            blocks[role] = _role_block(path, profile, role, table[role])
    return local_only, blocks


# --------------------------------------------------------------------------- overrides
def _split_override_key(key: str) -> tuple[str, str, str | None]:
    """'llm.url' -> ('llm', 'url', None); 'llm.options.max_tokens' -> ('llm', 'options', 'max_tokens')."""
    parts = key.split(".")
    if len(parts) == 2:
        role, k = parts
        opt = None
    elif len(parts) == 3 and parts[1] == "options":
        role, k, opt = parts[0], "options", parts[2]
    else:
        raise ConfigError(f"override {key!r}: expected '<role>.<key>' with key in {list(ROLE_KEYS)} "
                          f"or '<role>.options.<name>'")
    if role not in ROLES:
        raise ConfigError(f"override {key!r}: unknown role {role!r}; roles are {list(ROLES)}")
    if k not in ROLE_KEYS and k != "options":
        raise ConfigError(f"override {key!r}: unknown key {k!r}; allowed {list(ROLE_KEYS)} or options.<name>")
    if k == "options" and not opt:
        raise ConfigError(f"override {key!r}: options need a name, e.g. {role}.options.max_tokens")
    return role, k, opt


def _coerce_option(value: str) -> Any:
    """Env/override values are strings; make the obvious ones typed so options read like the TOML."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, str]:
    """UNSPOKEN_<ROLE>_<KEY> with KEY in PROVIDER/URL/MODEL/LABEL -> {'role.key': value}. Only exact role
    names count (UNSPOKEN_DECIDER_BIG_URL -> decider_big.url); other UNSPOKEN_* variables belong to other
    parts of the system and are left alone."""
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for role in ROLES:
        for key in ROLE_KEYS:
            name = f"{ENV_PREFIX}{role.upper()}_{key.upper()}"
            if name in env:
                out[f"{role}.{key}"] = env[name]
    return out


def apply_overrides(blocks: dict[str, dict[str, Any]], overrides: dict[str, str], source: str,
                    applied: list[str]) -> None:
    for key, value in overrides.items():
        if not isinstance(value, str):
            raise ConfigError(f"override {key!r} from {source}: value must be a string, got {type(value).__name__}")
        role, k, opt = _split_override_key(key)
        block = blocks.setdefault(role, {"provider": "none"})
        if k == "options":
            block.setdefault("options", {})[opt] = _coerce_option(value)
        else:
            block[k] = value
        applied.append(f"{source}: {key} = {value!r}")


# --------------------------------------------------------------------------- entry point
def resolve(profile: str | None = None, path: str | None = None, overrides: dict[str, str] | None = None,
            environ: dict[str, str] | None = None) -> ResolvedProfile:
    """The whole resolution: file + profile (+extends) + env overrides + dict overrides -> RoleSpecs."""
    name = resolve_profile_name(profile, environ)
    file = resolve_path(path, environ)
    doc = read_toml(file)
    local_only, blocks = _profile_blocks(doc, file, name)
    applied: list[str] = []
    apply_overrides(blocks, env_overrides(environ), "env", applied)
    apply_overrides(blocks, dict(overrides or {}), "overrides", applied)
    specs: dict[str, RoleSpec] = {}
    for role in ROLES:
        block = blocks.get(role, {"provider": "none"})
        specs[role] = RoleSpec(role=role, provider=str(block.get("provider") or "none"),
                               label=str(block.get("label") or ""), url=block.get("url") or None,
                               model=block.get("model") or None, options=dict(block.get("options") or {}))
    return ResolvedProfile(profile=name, path=str(file), local_only=local_only, specs=specs, applied=applied)
