"""Room screen: the full-screen page on the meeting-room TV (kiosk browser on 127.0.0.1) and its API.

    from unspoken.meeting.web import create_app
    app = create_app(runtime)          # runtime: unspoken.meeting.contracts.MeetingRuntimeAPI

Loopback only, no auth by design (see docs/SCREEN.md).
"""
from .server import create_app, is_loopback, require_loopback, serve, serve_async

__all__ = ["create_app", "is_loopback", "require_loopback", "serve", "serve_async"]
