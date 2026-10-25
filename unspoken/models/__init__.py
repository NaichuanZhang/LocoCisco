"""Model registry: every model the box runs has a ROLE; config/models.toml maps roles to providers per
PROFILE; code asks the registry for a role and never names a vendor. See docs/MODELS.md.

    from unspoken.models import load_registry
    reg = load_registry()                        # profile: arg > UNSPOKEN_PROFILE > "dev"
    reg = load_registry("gb10", overrides={"llm.url": "http://127.0.0.1:8000/v1"})

Check every role from the shell:  python -m unspoken.models check [--profile P] [--roles a,b] [--json]
"""
from __future__ import annotations

from ..contracts import ROLES, RemoteNotAllowed, Role, RoleDisabled, RoleHealth, RoleSpec
from .config import DEFAULT_PATH, DEFAULT_PROFILE, ENV_PATH, ENV_PROFILE, ConfigError, resolve
from .registry import ModelRegistry


def load_registry(profile: str | None = None, path: str | None = None,
                  overrides: dict[str, str] | None = None) -> ModelRegistry:
    """Build the registry for a profile.

    profile:   arg > UNSPOKEN_PROFILE env > "dev"
    path:      arg > UNSPOKEN_MODELS env > <repo>/config/models.toml
    overrides: {"llm.url": "...", "decider.provider": "fake", "llm.options.max_tokens": "300"}; these win
               over the env (UNSPOKEN_<ROLE>_<KEY>), which wins over the file.

    Raises ConfigError (bad file / unknown role or key), RemoteNotAllowed (a local_only profile points at a
    remote provider or host). Models are built lazily on first use, never here.
    """
    return ModelRegistry(resolve(profile=profile, path=path, overrides=overrides))


__all__ = ["load_registry", "ModelRegistry", "ConfigError", "RemoteNotAllowed", "RoleDisabled", "RoleHealth",
           "RoleSpec", "Role", "ROLES", "DEFAULT_PROFILE", "DEFAULT_PATH", "ENV_PROFILE", "ENV_PATH"]
