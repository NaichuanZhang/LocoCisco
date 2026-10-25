# Application source snapshot

The application was imported on 2026-10-25 from the local `build/lococisco-demo`
checkout at commit `ec41d8e93b2d419943e0745056b3b65ec03290e2`, including the
privacy fixes, recording frontend and autoplay. It builds on the earlier
`2fc471bc2b9373957210131d4f441f6d3370c530` checkpoint. The public Git history does
not import the original offline USB bundle's commit history. Existing public
presentation assets, design references, README and build specification are retained.

The snapshot contains the executable `unspoken` package, earlier `homedemo`
prototype, tests, examples, configuration templates, deployment scripts and docs.
The former source README is preserved as [LEGACY_SOURCE_README.md](LEGACY_SOURCE_README.md).
Runtime credentials, local agent/session notes, virtual environments, caches,
downloaded large models and generated meeting recordings are excluded.

Small vendored Silero VAD and YuNet models remain included so the existing
offline fixtures and resolver behavior are preserved. Their origins and licenses
are in [the sensor model sources](../unspoken/sensors/models/SOURCES.md).
The prepared demo resolves its other models from the existing external offline kit.
Three.js, furniture assets, fonts and other bundled dependencies retain their
existing license and attribution files. No new license is assigned to inherited
application code.

The application and presentation remain distinct: the reference dashboards and
slide diagrams contain explanatory samples, while `python -m unspoken.meeting.demo`
runs local inference on the official source video.
