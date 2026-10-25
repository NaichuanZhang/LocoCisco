"""Sensors: USB speakerphone + USB webcam -> MT.SEGMENT (voice label + text) and MT.SPEAKERS (which seat's
face is talking), live or from files, with privacy built in (no audio or video ever touches the disk; voices
and faces are forgotten at stop()).

Fixed interface (see docs/PLAN.md):
    async start_sensors(bus, registry, config, clock) -> SensorHandle   (SensorHandle.stop() is async)

In-process model factories for config/models.toml (`provider = "python:unspoken.sensors.<module>:<Factory>"`):
    vad.SileroVAD, vad.EnergyVAD, asr_local.FasterWhisperASR, diar.EmbeddingDiarizer, faces.YuNetFaces,
    asd.MotionASD. Each takes a RoleSpec and implements the role's protocol from unspoken.contracts.

Heavy imports (onnxruntime, cv2, faster_whisper) happen inside the factories, never at package import.
"""
from .config import load_sensors_config, seat_zones
from .pipeline import SensorHandle, SensorStats, start_sensors

__all__ = ["start_sensors", "SensorHandle", "SensorStats", "load_sensors_config", "seat_zones"]
