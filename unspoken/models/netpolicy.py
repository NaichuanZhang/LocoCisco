"""local_only profiles (the GB10): nothing may call a model outside this box or its LAN.

A URL is allowed when its host is loopback (127/8, ::1, "localhost"), an RFC 1918 address (10/8,
172.16/12, 192.168/16) or its IPv6 private counterpart (fc00::/7, RFC 4193), a *.local mDNS name, or the
conventional "inference.local" alias. Everything else, and the "bedrock" provider (always a remote AWS
endpoint), is refused with RemoteNotAllowed. Hostnames are NOT resolved: a public DNS name is refused
even if it happens to point at a LAN address, because the box has no business asking the internet.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from ..contracts import RemoteNotAllowed, RoleSpec

REMOTE_PROVIDERS: frozenset[str] = frozenset({"bedrock"})
LOCAL_HOSTNAMES: frozenset[str] = frozenset({"localhost", "inference.local"})
LAN_SUFFIX = ".local"
RFC1918 = (ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"),
           ipaddress.ip_network("192.168.0.0/16"))
IPV6_PRIVATE = ipaddress.ip_network("fc00::/7")


def host_is_local(host: str | None) -> bool:
    if not host:
        return False
    host = host.strip("[]").lower().rstrip(".")
    if host in LOCAL_HOSTNAMES or host.endswith(LAN_SUFFIX):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in RFC1918)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return host_is_local(str(ip.ipv4_mapped))
    return ip in IPV6_PRIVATE


def url_is_local(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return host_is_local(parts.hostname)


def assert_local(spec: RoleSpec, profile: str) -> None:
    """Raise RemoteNotAllowed when a local_only profile names a remote provider or URL for this role."""
    if spec.provider == "none":
        return
    if spec.provider in REMOTE_PROVIDERS:
        raise RemoteNotAllowed(f"profile {profile!r} is local_only: role {spec.role!r} uses provider "
                               f"{spec.provider!r}, which always calls a remote service")
    if spec.url and not url_is_local(spec.url):
        raise RemoteNotAllowed(f"profile {profile!r} is local_only: role {spec.role!r} url {spec.url!r} is not "
                               f"loopback, RFC 1918, *.local or inference.local")
