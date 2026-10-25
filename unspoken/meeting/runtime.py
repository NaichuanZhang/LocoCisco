"""Composition root: one bus, the meeting state, the agents, the registry, devices, notifier, docs and the
scenario runner. Implements contracts.MeetingRuntimeAPI for the room screen and the MCP server.

Modes
  sim     the simulator turns scenario steps into sensor events (no hardware, any machine)
  live    unspoken.sensors.start_sensors publishes audio.segment / video.speakers from the mic and camera
  replay  a recorded bus trace (JSONL of Event.to_json()) is replayed at settings.speed

Other workstreams' packages are optional today: the registry falls back to meeting.fakes.LocalRegistry,
devices to fakes, the Notifier to the console. Order of bus subscriptions matters for reset(): the sim
and the scenario runner subscribe first, the agents last, so reset rebuilds only the agents.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import time
import tomllib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

from ..core.bus import Bus
from ..core.events import Event
from .agents import AGENTS, start_agents
from .base import MeetingState, local_llm_enabled
from .config import REPO_ROOT, load_config, settings_from
from .contracts import MT, SENSOR_TOPICS, SPOKEN_TOPICS, ChatCommandData, MeetingSettings, PrivacyData, PrivacyState
from .docs import DocIndex, answer_from
from .fakes import FakeNotifier, LocalRegistry, MeteredNotifier, fake_devices
from .recap import recap_markdown
from .sim import World

log = logging.getLogger("unspoken.meeting.runtime")
REPLAY_TOPICS = set(SENSOR_TOPICS) | {MT.CLOCK, MT.STATUS}


def _optional(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except Exception:    # ImportError, or a workstream's module failing on this machine (no torch, no camera)
        return None


def _toml(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "config" / name
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_registry(profile: str | None, overrides: dict[str, str] | None = None) -> Any:
    """unspoken.models.load_registry when the models workstream's package exists, else LocalRegistry."""
    models = _optional("unspoken.models")
    if models is not None and hasattr(models, "load_registry"):
        # For the meeting room, "fake" means its keyword decider with no LLM: profile "sim" in config/models.toml.
        resolved = "sim" if profile == "fake" else profile
        try:
            return models.load_registry(profile=resolved, overrides=overrides)
        except Exception as exc:
            log.warning("unspoken.models.load_registry failed (%s); using the fallback registry", exc)
    return LocalRegistry(profile or "fake", overrides=overrides)


def build_devices(config: dict[str, Any], *, fake: bool = False) -> dict[str, Any]:
    if not fake:
        devices = _optional("unspoken.devices")
        if devices is not None and hasattr(devices, "load_devices"):
            try:
                return dict(devices.load_devices(_toml("room.toml")))
            except Exception as exc:
                log.warning("unspoken.devices.load_devices failed (%s); using fake devices", exc)
    return fake_devices(config)


def build_notifier(on_egress: Callable[[str, str, int], None], *, fake: bool = False, echo: bool = True) -> Any:
    """unspoken.bridge.make_notifier(load_bridge_config(), on_egress) when the bridge package exists (transport from
    config/bridge.toml [notifier] or UNSPOKEN_NOTIFIER=console|openclaw|discord-rest; it meters its own egress),
    else the recording console fake behind the egress meter."""
    if not fake:
        bridge = _optional("unspoken.bridge")
        if bridge is not None and hasattr(bridge, "make_notifier"):
            try:
                cfg = bridge.load_bridge_config() if hasattr(bridge, "load_bridge_config") else _toml("bridge.toml")
                return bridge.make_notifier(cfg, on_egress=on_egress)
            except Exception as exc:
                log.warning("unspoken.bridge.make_notifier failed (%s); using the console notifier", exc)
    return MeteredNotifier(FakeNotifier(echo=echo), on_egress=on_egress)


class MeetingRuntime:
    """contracts.MeetingRuntimeAPI."""

    bus_type = Bus

    def __init__(self, *, profile: str | None = None, mode: str = "sim", registry: Any = None,
                 config: dict[str, Any] | None = None, settings: MeetingSettings | None = None,
                 devices: dict[str, Any] | None = None, notifier: Any = None, fake_io: bool = False, echo: bool = True,
                 trace_path: str | None = None, tick_seconds: float = 1.0, replay_path: str | None = None,
                 overrides: dict[str, str] | None = None) -> None:
        self.profile = profile or "fake"
        self.config = config or load_config()
        self.settings = settings or settings_from(self.config)
        self.settings.mode = mode  # type: ignore[assignment]
        self.registry = registry if registry is not None else build_registry(self.profile, overrides)
        self.bus = self.bus_type(trace_path=trace_path)
        self.state = MeetingState()
        # Nothing said while notes are paused or off the record is kept in the bus history or trace file.
        self.bus.retain = lambda e: e.topic not in SPOKEN_TOPICS or self.state.privacy.get("state") == "on"
        self.devices = devices if devices is not None else build_devices(self.config, fake=fake_io or self.profile == "fake")
        self.notifier = notifier if notifier is not None else build_notifier(self._on_egress, fake=fake_io or self.profile == "fake", echo=echo)
        embedder = None
        try:
            if self.registry.enabled("embed"):
                embedder = self.registry.embedder()
        except Exception:
            embedder = None
        self.docs = DocIndex(str(self.config["meeting"]["docs_dir"]), embedder).load()
        self.world = World(self.bus, absent_after=float(self.config["meeting"].get("absent_after_seconds", 3.0)))
        self.replay_path = replay_path
        self.agents: list = []
        self.models_health: list[dict[str, Any]] = []
        self._agent_sub_index = 0
        self._tick_seconds = tick_seconds
        self._tick_task: asyncio.Task | None = None
        self._scenario_task: asyncio.Task | None = None
        self._sensors: Any = None
        self._runner: Any = None
        self._lock = asyncio.Lock()
        self._live_started: float | None = None

    # ------------------------------------------------------------------ wiring
    @property
    def mode(self) -> str:
        return str(self.settings.mode)

    def _llm_factory(self) -> Callable[[], Any] | None:
        try:
            if local_llm_enabled(self.registry):
                return lambda: self.registry.strands_model("llm")
        except Exception:
            return None
        return None

    def _on_egress(self, dest: str, kind: str, nbytes: int) -> None:
        e = self.state.egress
        e.update(dest=dest, kind=kind, bytes=int(nbytes), total_bytes=int(e.get("total_bytes", 0)) + int(nbytes),
                 audio_bytes_out=0, video_bytes_out=0)

    def _start_agents(self) -> None:
        self._agent_sub_index = self.bus.subscription_count()
        self.agents = start_agents(self.bus, self.state, self.settings, self.config, registry=self.registry,
                                   devices=self.devices, notifier=self.notifier, docs=self.docs, llm_factory=self._llm_factory())

    def agent(self, name: str) -> Any:
        return next(a for a in self.agents if a.name == name)

    async def start(self, *, ticking: bool = True) -> None:
        from .scenarios import ScenarioRunner

        self._runner = ScenarioRunner(self.bus, self.world, self.settings, snapshot=self.snapshot)
        self._start_agents()
        await self.refresh_models()
        if self.mode == "live":
            await self._start_live()
        elif self.mode == "replay":
            await self._start_replay()
        if ticking and self._tick_seconds > 0:
            self._tick_task = asyncio.create_task(self._tick_loop(), name="meeting.tick")
        await self.bus.drain(60)

    async def refresh_models(self) -> list[dict[str, Any]]:
        try:
            health = await self.registry.check()
            self.models_health = [asdict(h) if hasattr(h, "__dataclass_fields__") else dict(h) for h in health]
        except Exception as exc:
            self.models_health = [{"role": "registry", "ok": False, "detail": str(exc)}]
        return self.models_health

    async def _start_live(self, *, replay: dict[str, Any] | None = None) -> None:
        """Live: unspoken.sensors.start_sensors(bus, registry, load_sensors_config(), clock). Replay of a recording
        goes through the same pipeline with config["audio"]["path"] / config["video"]["path"] set."""
        sensors = _optional("unspoken.sensors")
        m = self.config["meeting"]
        await self.bus.publish(Event(MT.STATUS, {"state": "started", "title": m.get("title"), "owner": m.get("owner"),
                                                 "agenda": list(m.get("agenda") or []), "minutes": m.get("minutes")},
                                     source="runtime", ts=0.0))
        self._live_started = time.monotonic()
        if sensors is None or not hasattr(sensors, "start_sensors"):
            await self.bus.publish(Event(MT.LOG, {"agent": "runtime", "level": "warn",
                                                  "text": "unspoken.sensors is not available: live mode has no sensors (chat and the API still work)"},
                                         source="runtime"))
            return
        cfg = sensors.load_sensors_config() if hasattr(sensors, "load_sensors_config") else _toml("sensors.toml")
        if replay:
            cfg = {**cfg, **replay}
        clock = lambda: time.monotonic() - (self._live_started or time.monotonic())  # noqa: E731
        self._sensors = await sensors.start_sensors(self.bus, self.registry, cfg, clock)

    async def _start_replay(self) -> None:
        """A .wav/.mp4 recording replays through the real sensors (VAD/ASR/diarization/faces); a .jsonl bus trace
        republishes the recorded sensor events."""
        if not self.replay_path:
            raise ValueError("replay mode needs replay_path (a .wav/.mp4 recording or a bus trace .jsonl)")
        path = Path(self.replay_path)
        if path.suffix.lower() in (".wav", ".mp4", ".mkv", ".mov", ".mp3", ".flac"):
            replay: dict[str, Any] = {"replay": {"realtime": True, "speed": float(self.settings.speed)}}
            if path.suffix.lower() in (".mp4", ".mkv", ".mov"):
                replay["video"] = {"path": str(path)}
            else:
                replay["audio"] = {"path": str(path)}
            await self._start_live(replay=replay)
            return
        self._scenario_task = asyncio.create_task(self._replay(path), name="meeting.replay")

    async def _replay(self, path: Path) -> None:
        last: float | None = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if raw.get("topic") not in REPLAY_TOPICS:
                    continue
                ts = float(raw.get("ts") or 0.0)
                if last is not None and ts > last:
                    await asyncio.sleep((ts - last) / max(0.01, float(self.settings.speed)))
                last = ts
                await self.bus.publish(Event(raw["topic"], raw["data"], source=f"replay.{raw.get('source', '?')}", ts=ts))

    async def _tick_loop(self) -> None:
        tick = float(self.config["meeting"].get("clock_tick_seconds", 10.0))
        last_clock = 0.0
        while True:
            await asyncio.sleep(self._tick_seconds)
            if self._lock.locked():
                continue
            try:
                if self.mode == "sim" and self.world.started and not (self._runner and self._runner.running):
                    await self.world.tick(self._tick_seconds * float(self.settings.speed))
                if self.mode == "live" and self._live_started is not None:
                    now = time.monotonic() - self._live_started
                    if now - last_clock >= tick:
                        last_clock = now
                        await self.bus.publish(Event(MT.CLOCK, {"t": now, "minutes_left": self._minutes_left(now)},
                                                     source="runtime", ts=now))
            except Exception:
                log.exception("tick failed")

    def _minutes_left(self, now: float) -> float | None:
        minutes = self.config["meeting"].get("minutes")
        if minutes is None:
            return self.state.minutes_left
        return max(0.0, float(minutes) - now / 60.0)

    async def close(self) -> None:
        await self.stop_scenario()
        if self._tick_task:
            self._tick_task.cancel()
        await self.stop_sensors()
        await self.bus.cancel_pending()
        devices_mod = _optional("unspoken.devices")
        if devices_mod is not None and hasattr(devices_mod, "close_devices"):
            try:
                await devices_mod.close_devices(self.devices)
            except Exception:
                log.exception("close_devices failed")
        if hasattr(self.registry, "aclose"):
            try:
                await self.registry.aclose()
            except Exception:
                pass

    async def stop_sensors(self) -> None:
        """Stop the live sensors: SensorHandle.stop() forgets every voice and face (privacy)."""
        if self._sensors is not None and hasattr(self._sensors, "stop"):
            try:
                await self._sensors.stop()
            except Exception:
                log.exception("sensors.stop failed")
        self._sensors = None

    def sensors_xray(self) -> dict[str, Any] | None:
        h = self._sensors
        if h is None:
            return None
        out: dict[str, Any] = {"models": dict(getattr(h, "models", {}) or {})}
        stats = getattr(h, "stats", None)
        if stats is not None and hasattr(stats, "summary"):
            try:
                out["stats"] = stats.summary()
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------ MeetingRuntimeAPI
    def snapshot(self) -> dict[str, Any]:
        s = self.state
        return {
            "status": dict(s.status), "privacy": dict(s.privacy), "participants": s.participants(),
            "screen": dict(s.screen), "items": [dict(i) for i in s.items.values()],
            "devices": [dict(d) for d in s.devices.values()], "egress": dict(s.egress), "models": list(self.models_health),
            "settings": asdict(self.settings), "mode": self.mode, "profile": self.profile,
            "agenda": list(s.agenda), "minutes_left": s.minutes_left, "clock": s.clock,
            "scenario": (self._runner.current or {}).get("id") if self._runner else None,
            "sensors": self.sensors_xray(),
            "agents": [{**a, "busy": bool(getattr(self._agent_or_none(a["name"]), "busy", False))} for a in AGENTS],
        }

    def _agent_or_none(self, name: str) -> Any:
        return next((a for a in self.agents if a.name == name), None)

    def transcript(self, since: float | None = None) -> list[dict[str, Any]]:
        return [dict(u) for u in self.state.transcript if since is None or float(u.get("start", 0.0)) >= since]

    def items(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [dict(i) for i in self.state.items.values() if kind is None or i.get("kind") == kind]

    async def search_docs(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        return await self.docs.search(query, k)

    async def recap(self) -> str:
        return await recap_markdown(self.state, self._llm_factory())

    async def chat(self, command: ChatCommandData) -> dict[str, Any]:
        cmd = str(command.get("command") or "")
        data = {"command": cmd, "user": command.get("user") or "someone", "args": dict(command.get("args") or {}),
                "text": command.get("text") or ""}
        await self.bus.publish(Event(MT.CHAT, data, source="chat", ts=self.state.clock))
        try:
            await self.bus.drain(timeout=30)
        except TimeoutError:
            pass
        if cmd == "here":
            return {"ok": True, "text": f"Welcome, {data['args'].get('name') or data['user']}. You're checked in."}
        if cmd == "recap":
            return {"ok": True, "text": await self.recap()}
        if cmd == "ask":
            hits = await self.search_docs(data["text"] or str(data["args"].get("text") or ""))
            return {"ok": True, "text": answer_from(hits, data["text"])}
        if cmd in ("offrecord", "resume"):
            return {"ok": True, "text": f"Notes are {self.state.privacy.get('state')}."}
        if cmd == "undo":
            return {"ok": True, "text": "Removed."}
        if cmd in ("approve", "deny"):
            return {"ok": True, "text": f"{cmd.capitalize()}d."}
        if cmd == "agenda":
            return {"ok": True, "text": f"Agenda: {', '.join(self.state.agenda) or 'empty'}."}
        return {"ok": False, "text": f"unknown command {cmd!r}"}

    async def set_privacy(self, state: PrivacyState, reason: str, by: str = "user") -> PrivacyData:
        reflex = self.agent("reflex")
        data = await reflex.set_privacy(state, reason, by, None, wipe=(state == "off_record"))
        await self.bus.drain(30)
        return data

    def scenarios(self) -> list[dict[str, Any]]:
        from .scenarios import load_scenarios, summary

        return [summary(s) for s in load_scenarios()]

    async def run_scenario(self, scenario_id: str) -> None:
        from .scenarios import get

        scenario = get(scenario_id)
        await self.reset()
        self._scenario_task = asyncio.create_task(self._runner.run(scenario), name=f"scenario:{scenario_id}")

    async def stop_scenario(self) -> None:
        if self._runner is not None and self._scenario_task and not self._scenario_task.done():
            await self._runner.stop()
            self._scenario_task.cancel()
            await asyncio.gather(self._scenario_task, return_exceptions=True)
        self._scenario_task = None

    async def reset(self) -> None:
        async with self._lock:
            await self.stop_scenario()
            await self.bus.cancel_pending()
            self.bus.unsubscribe_from(self._agent_sub_index)
            self.state = MeetingState()
            self._start_agents()
            await self.world.reset()
            await self.bus.drain(60)
            await self.bus.publish(Event(MT.LOG, {"agent": "runtime", "level": "info", "text": "Meeting reset. Memory wiped."},
                                         source="runtime", ts=0.0))

    async def update_settings(self, **changes: Any) -> MeetingSettings:
        known = {f.name for f in fields(MeetingSettings)}
        bad = set(changes) - known
        if bad:
            raise ValueError(f"unknown settings: {sorted(bad)}")
        for k, v in changes.items():
            setattr(self.settings, k, v)
        await self.bus.publish(Event(MT.LOG, {"agent": "runtime", "level": "info", "text": f"Settings changed: {changes}"},
                                     source="runtime", ts=self.state.clock))
        return self.settings

    # ------------------------------------------------------------------ extras used by the CLI / tests
    async def step(self, step: dict[str, Any]) -> None:
        await self.world.step(step)

    def settings_dict(self) -> dict[str, Any]:
        return asdict(self.settings)
