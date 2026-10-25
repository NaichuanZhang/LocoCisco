# DEVICES workstream report (ws/devices)

Branch `ws/devices`, worktree `unspoken-wt/devices`. Time spent: ~2 h 15 min. STOP condition met: the CLI
drives the fake WiZ bulb end to end and the whole suite passes (464 passed = 383 existing + 81 new).

## 1. What you built

| Path | What |
|---|---|
| `unspoken/devices/__init__.py` | `load_devices(config) -> dict[str, Device]` from room.toml; missing / unknown driver / no host / bad values -> unhealthy `FakeLight`/`FakePlug` (never crashes); `load_room_config()` (never raises; `UNSPOKEN_ROOM_CONFIG` override); `probe_devices()`, `close_devices()` |
| `unspoken/devices/scenes.py` | the 9 scenes -> WiZ pilots (`privacy` soft green, `paused` amber, `off_record` red, `talk`, `slides`, `break`, `bright`, `dim`, `off`), colour / brightness helpers, pilot <-> state, `merge_pilot` (what the bulb keeps) |
| `unspoken/devices/wiz.py` | own WiZ local-UDP client (setPilot / getPilot / getSystemConfig / registration): 1 s timeout, 2 retries, stale-reply skipping, one try once known down; `WizLight` Device with leading+trailing debounce and the never-raise contract; `discover_wiz()` broadcast discovery |
| `unspoken/devices/kasa_plug.py` | `KasaPlug` Device on python-kasa (lazy import; IOT and KLAP/SMART handled by the library), credentials only from `KASA_USERNAME`/`KASA_PASSWORD`, retries, never raises; `discover_kasa()` flags plugs that need credentials |
| `unspoken/devices/fakes.py` | `FakeLight`, `FakePlug` (in memory); `FakeWizBulb` UDP server speaking the real WiZ protocol (drop / delay / reject knobs); `FakeKasaIotPlug` loopback server speaking the real legacy Kasa protocol (UDP discovery + TCP XOR) so python-kasa's own client is tested end to end; `FakeKasaDevice` for the connector seam |
| `unspoken/devices/__main__.py` | CLI: `discover` (prints a pasteable room.toml block), `list`, `set lamp scene off_record`, `set fan off`, `scenes`, `fake-bulb`; `--json`; exit 1 when a device is down |
| `config/room.toml` | `[devices.lamp]` light / `[devices.fan]` plug, `driver = "fake"` by default with the real-driver lines commented |
| `requirements/devices.txt` | `python-kasa==0.11.0.1` |
| `docs/DEVICES.md` | travel-router setup, pairing once, discovery and test commands, room.toml, scenes, Kasa buyer's list (works / needs credentials / avoid), timing and failure behaviour, deps + aarch64, risk |
| `tests/devices/` | 82 tests (81 run, 1 live-only): scenes, WiZ wire format and client, WizLight end to end on the fake bulb (every scene, set, debounce, timeouts, recovery, rejection, state refresh), KasaPlug (seam + real python-kasa on the loopback fake), load_devices fallbacks, CLI against a fake-bulb subprocess |

## 2. How to run it

```bash
cd unspoken-wt/devices      # or the integration checkout after merge
UV_CACHE_DIR=... scripts/dev-venv.sh && uv pip install --python .venv/bin/python -r requirements/devices.txt

python -m unspoken.devices discover                       # on the travel router: WiZ + Kasa -> room.toml block
python -m unspoken.devices list                           # live state + health, exit 1 if anything is DOWN
python -m unspoken.devices set lamp scene off_record      # also: privacy paused talk slides break bright dim off
python -m unspoken.devices set lamp set brightness=40 color=#ff8800
python -m unspoken.devices set fan off
python -m unspoken.devices fake-bulb --port 38899         # a bulb on 127.0.0.1 for rehearsal without hardware
```
From code: `devices = load_devices(load_room_config()); await devices["lamp"].apply("scene", {"scene": "privacy"})`.
Kasa plugs on KLAP/SMART: `export KASA_USERNAME=... KASA_PASSWORD=...` first (never in a file).

## 3. Verification

- `.venv/bin/python -m pytest -q` (whole suite, this worktree): **464 passed, 3 skipped in 21.05 s**
  (383 existing + 81 new; skips = 2 existing live tests + my live-bulb test). `pytest -q tests/devices`:
  81 passed, 1 skipped in 5.5 s. No model servers, no hardware.
- Real python-kasa client against `FakeKasaIotPlug` (legacy protocol on loopback): discovery over UDP,
  `update`, `turn_on`, `turn_off`, mid-session TCP drop -> `healthy: false` with python-kasa's error,
  reconnect -> healthy again (`tests/devices/test_kasa_plug.py::test_real_python_kasa_client_drives_the_fake_legacy_plug`).
- CLI against `python -m unspoken.devices fake-bulb --port 38899` with a room.toml pointing at
  127.0.0.1 (`UNSPOKEN_ROOM_CONFIG`): `discover --no-kasa --broadcast 127.0.0.1` found the bulb and
  printed `[devices.lamp] kind="light" driver="wiz" host="127.0.0.1"` + mac/module/fw; `set lamp scene
  off_record` -> `ok on=True brightness=70 scene=off_record color=#ff0000` (whole command 0.10 s wall,
  Python start-up included); `set lamp scene privacy`, `set lamp set brightness=40 color=#ff8800`,
  `set fan off` -> exit 0; `list` -> both `ok`, exit 0; the bulb's log showed exactly the expected
  pilots. Bulb killed: `list` -> `lamp ... DOWN ... error: getPilot ... failed after 3 attempt(s):
  ConnectionRefusedError`, exit 1, 0.1 s; `set lamp scene talk` -> DOWN, exit 1. Default
  `config/room.toml` (fakes): `list` -> both `ok`, exit 0.
- Measured on loopback: setPilot round trip 0.4 ms; 8 concurrent scene commands -> 1-2 datagrams,
  final bulb scene = newest command; 2 dropped datagrams with a 0.3 s timeout -> success after 652 ms;
  bulb gone (all drops) -> `healthy: false` after 3 x timeout, then one attempt per command (301 ms at
  0.3 s); closed UDP port -> fails in 1 ms (ICMP). python-kasa first contact with the fake plug 190 ms,
  later on/off 0-1 ms. Real `discover --timeout 1` on this desktop (no devices): empty, well-formed, 1.35 s.
- Lint: `ruff check --select E,F,W,B,UP` clean except the repo-wide `typing.Callable` import style.

## 4. Not verifiable here; how to verify on the GB10 / travel router

- No WiZ bulb and no Kasa plug on this desktop: colour rendering of the pilots, real Wi-Fi latency and
  loss, whether the bought plug speaks IOT or KLAP. On the day: join the travel router, run
  `python -m unspoken.devices discover`, paste the block into `config/room.toml`, then
  `python -m unspoken.devices list` (both `ok`) and `UNSPOKEN_LIVE=1 UNSPOKEN_WIZ_HOST=<ip>
  .venv/bin/python -m pytest -q tests/devices -k live` (cycles privacy / paused / off_record / talk on
  the real bulb). If `discover` prints `NEEDS KASA_USERNAME/KASA_PASSWORD`, export them and re-run `list`.
- Broadcast discovery depends on the router passing 255.255.255.255; `--broadcast 192.168.8.255`
  (the subnet broadcast) is the fallback, and the bulb's IP can simply be read from the router's DHCP table.
- aarch64 install: `uv pip install -r requirements/devices.txt` on DGX OS; wheel availability was
  checked on PyPI (section 5) but not installed on Arm here.

## 5. Contract change requests, deps, config for other owners

Contract (docs only, no field changes):
1. `unspoken/contracts.py` `Device.apply` docstring lists light scenes `talk|slides|break|privacy|off_record`;
   the room also uses `paused`, `bright`, `dim`, `off` (`ACTIONS` already names `bright|dim`). Please list all
   nine: `talk|slides|break|privacy|paused|off_record|bright|dim|off`.
2. `FaceTrack.seat` says "seat id from config/room.toml", but `_common.md` puts seat zones in
   `config/sensors.toml` and `config/room.toml` (devices-owned) holds devices only. Please point the
   comment at `config/sensors.toml` (or tell me to host a `[seats]` table in room.toml).

Deps added: `python-kasa==0.11.0.1` (pure Python). Resolved tree 2026-10-24: aiohttp 3.14.5,
aiosignal, aiohappyeyeballs, asyncclick, attrs, frozenlist, idna, mashumaro, multidict, propcache, yarl
(all universal wheels); cryptography 50.0.2 and cffi 2.1.1 (manylinux aarch64 cp312/abi3 wheels on PyPI).
Nothing lacks an aarch64 wheel. WiZ needs no dependency (pywizlight not used).

For the meeting workstream (actuator): `set_light {scene}` -> `devices["lamp"].apply("scene", {"scene": scene})`;
`set_fan {on}` -> `devices["fan"].apply("on" if on else "off")`; privacy state -> scene `privacy` (on),
`paused`, `off_record`. Every `apply()`/`state()` result is a dict ready for `EnvData.state`
(`MT.ENV {"device_id", "kind", "state"}`); `state["healthy"]` is False when the appliance is unreachable
and `state["error"]` says why. Call `probe_devices(devices)` once at start for the snapshot. Devices
hold a UDP/TCP socket: call `close_devices(devices)` on shutdown. No config blocks needed from others;
secrets: `KASA_USERNAME`, `KASA_PASSWORD` (env only, optional).

## 6. Open risks

- A plug bought on the day may be KLAP/SMART (KP125M, EP25 hw 2.6, every Tapo): works, but needs the
  TP-Link account in env; a plug that auto-updated to KLAP overnight shows up as DOWN until the vars are set.
- WiZ bulbs are password-less on their Wi-Fi: only on our travel router (docs/DEVICES.md section 1).
- Debounce intentionally drops intermediate scene changes inside 150 ms; a scenario that wants two visible
  flashes must space them out.
- `discover_single` uses a 1 s discovery window; on a congested Wi-Fi set `timeout = 2` on the fan entry.
- Not tested on real hardware (section 4).
