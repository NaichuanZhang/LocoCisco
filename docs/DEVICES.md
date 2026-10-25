# Devices: the lamp and the fan

LocoCisco drives two appliances in the room, both over the local network and never through a vendor cloud:

| id | what | protocol | code |
|---|---|---|---|
| `lamp` | a WiZ colour bulb in a desk lamp: the **privacy lamp** (green = notes on, amber = paused, red = off the record) and room scenes (talk, slides, break, bright, dim) | WiZ local UDP JSON, port 38899 (`setPilot` / `getPilot`), no password | `unspoken/devices/wiz.py` (own ~200-line client, no dependency) |
| `fan` | a TP-Link Kasa smart plug with a small fan: stands in for the AC | Kasa IOT (TCP 9999) or KLAP/SMART (HTTP 80) via `python-kasa` | `unspoken/devices/kasa_plug.py` |

Both implement `unspoken.contracts.Device` (`apply`, `state`, `healthy`) and **never raise**: a device
that is off the network answers with its last known state, `"healthy": false` and an `"error"` line, and
tries again on the next command. `load_devices()` turns anything missing or misconfigured into an
unhealthy in-memory fake, so the room brain always has a `lamp` and a `fan` and the demo never crashes
on hardware.

## 1. Network: our own travel router, never the venue Wi-Fi

WiZ bulbs take commands from anyone on their Wi-Fi: there is no password on the local protocol
(see the protocol notes in [pywizlight](https://github.com/sbidy/pywizlight)). Kasa plugs on the
legacy protocol are the same. So:

1. Bring a small travel router (any 2.4 GHz-capable router; the bulb and the plug are 2.4 GHz only).
   Give it a WPA2 network nobody else knows (e.g. `unspoken-room`). The GB10 joins it too (Wi-Fi or a
   LAN cable to the router). The router does not need an uplink for the devices to work.
2. **Pair each device once at home with its phone app** (WiZ app for the bulb, Kasa app for the plug),
   onto the travel router's Wi-Fi, so they remember that network. Pairing needs the phone on the same
   Wi-Fi and, for Kasa, a TP-Link account. After that the apps are not needed: LocoCisco talks to the
   devices directly.
3. In the router, give both devices DHCP reservations so their IPs do not move between home and the venue.
   Write the IPs into `config/room.toml` (or run `discover` on the day, below).
4. Optional and recommended: block the devices' internet access in the router (parental control /
   firewall). The bulb and a legacy-protocol plug keep working; a KLAP plug keeps working too as long as
   its credentials are set (section 4). WiZ auto-updates and cloud chatter stop.

## 2. Commands

```bash
# one-off setup in the worktree (deps for this workstream only)
uv pip install --python .venv/bin/python -r requirements/devices.txt

# on the travel router's Wi-Fi: find the bulb and the plug, get a room.toml block to paste
python -m unspoken.devices discover                 # add --timeout 5 on a slow network
python -m unspoken.devices discover --broadcast 192.168.8.255   # if the default broadcast is filtered

# check: every device in room.toml with live state and health (exit 1 when anything is DOWN)
python -m unspoken.devices list

# drive them
python -m unspoken.devices set lamp scene off_record
python -m unspoken.devices set lamp scene privacy
python -m unspoken.devices set lamp set brightness=40 color=#ff8800
python -m unspoken.devices set lamp off
python -m unspoken.devices set fan on
python -m unspoken.devices set fan off
python -m unspoken.devices scenes                   # the scene table
python -m unspoken.devices --json list              # machine-readable; also --json for set / discover

# no hardware? run a fake WiZ bulb on 127.0.0.1 and point room.toml at it
python -m unspoken.devices fake-bulb --port 38899   # terminal 1 (prints every request it gets)
UNSPOKEN_ROOM_CONFIG=/path/room-local.toml python -m unspoken.devices set lamp scene off_record   # terminal 2

# tests (no hardware, no servers)
.venv/bin/python -m pytest -q tests/devices
# against a real bulb (sets four scenes, 0.7 s apart, ends on "talk")
UNSPOKEN_LIVE=1 UNSPOKEN_WIZ_HOST=192.168.8.21 .venv/bin/python -m pytest -q tests/devices -k live
```

`--config PATH` or `UNSPOKEN_ROOM_CONFIG=PATH` selects another room.toml.

## 3. config/room.toml

```toml
[devices.lamp]
kind = "light"
driver = "wiz"            # "wiz" | "fake"
host = "192.168.8.21"     # from `discover`
# port = 38899            # optional (default)
# timeout = 1.0           # s per datagram (default)
# retries = 2             # extra datagrams after a timeout (default)
# debounce_s = 0.15       # coalesce bursts: first command goes at once, the newest after the window

[devices.fan]
kind = "plug"
driver = "kasa"           # "kasa" | "fake"
host = "192.168.8.22"
# port = 9999             # only for a legacy plug on a non-default port
# timeout = 1.0
# retries = 2
```

`driver = "fake"` gives an in-memory device (the repo default, so tests and the simulator run anywhere).
A `fake` entry may carry `healthy = false` and `error = "..."` to rehearse the failure path on screen.

## 4. Kasa plugs: buy one that still speaks the local protocol without an account

python-kasa supports two Kasa generations. The **legacy IOT protocol** (TCP 9999, XOR) needs no
credentials at all. Newer hardware, and some firmware updates, use **KLAP / SMART** (HTTP 80, encrypted
handshake), which is still LAN-local but authenticates with the TP-Link account the plug was set up
with: then `KASA_USERNAME` and `KASA_PASSWORD` must be in the environment (never in a file in this
repo; python-kasa's `kasa` CLI reads the same two variables). Source: python-kasa
[Protocols and Transports](https://python-kasa.readthedocs.io/en/stable/topics.html#protocols-and-transports)
and the [supported devices list](https://github.com/python-kasa/python-kasa/blob/master/SUPPORTED.md)
(entries marked "requires authentication"), checked 2026-10-24.

**Buy / bring (no credentials, legacy protocol):** Kasa **EP10**, **HS103**, **HS105**, **KP105**,
**KP115** (energy meter), **KP125** (hw 1.0), **HS100** hw 1.0/2.0, **HS110**, **KP100**, **KP401**
(outdoor), **EP25** hw 1.0; power strips **HS300**, **KP303**, **KP400**, **EP40**. Prefer EP10 / HS103 /
KP115: cheap, US plug, plenty of python-kasa fixtures.

**Works only with KASA_USERNAME / KASA_PASSWORD (KLAP or SMART):** **KP125M** (Matter), **EP40M**, **EP25**
hw 2.6, **HS100** hw 4.1 (UK), every **Tapo** plug (P100, P105, P110, P115, P125M, P135, TP15, ...).
Fine for us if you have the account; a plug that has never been joined to the TP-Link cloud may even
answer without credentials (python-kasa docs), but do not count on it.

**Avoid:** anything that only lists Matter / HomeKit and no Kasa or Tapo app support (no LAN API
python-kasa knows), and any plug whose box says "Tapo" when you want the no-account path.

**Watch out:** TP-Link ships protocol changes in automatic firmware updates (that is how the UK HS100
moved to KLAP). Turn off auto-update in the Kasa app after pairing, or block the plug's internet on the
router (section 1), and run `python -m unspoken.devices discover`: it prints `NEEDS KASA_USERNAME/...`
next to a plug that now wants credentials.

## 5. Scenes

| scene | pilot sent to the bulb | when |
|---|---|---|
| `privacy` | soft green `#3cc85a`, 50 % | notes are ON (the privacy lamp's default while recording) |
| `paused` | amber `#ffa000`, 60 % | notes paused |
| `off_record` | red `#ff0000`, 70 % | someone said "off the record" |
| `talk` | white 4000 K, 80 % | people talking |
| `slides` | white 2700 K, 25 % | dim and warm so the screen reads |
| `break` | white 2200 K, 50 % | break |
| `bright` | white 5000 K, 100 % | |
| `dim` | white 2700 K, 15 % | |
| `off` | state off | |

`set` takes `brightness` 0-100 (0 = off; 1-9 become 10, the bulb's floor) and/or `color = "#rrggbb"`.
State dicts (`state()` and every `apply()` result) look like
`{"on": true, "brightness": 70, "color": "#ff0000", "temp": null, "scene": "off_record", "healthy": true, "error": "", "driver": "wiz", "host": "192.168.8.21"}`
for the lamp and `{"on": false, "healthy": true, "error": "", "driver": "kasa", "host": "...", "alias": "fan", "model": "HS103"}`
for the fan; they drop straight into `EnvData.state` (`MT.ENV`).

## 6. Timing and failure behaviour

- WiZ: one datagram per command, **1 s timeout, 2 retries** while the bulb is believed healthy (worst
  case ~3 s), **one try** once it is known down (~1 s, so a dead lamp does not stall the room). Loopback
  / ICMP-unreachable hosts fail in milliseconds. Bulb-side rejections ("Invalid params") are not retried
  and keep the bulb healthy. `state()` is served from cache for 2 s, then re-read from the bulb
  (so a change made in the WiZ app shows up).
- Debounce: bursts (the decider flapping, a scenario firing several scene changes) send the first
  command at once and only the **newest** after `debounce_s` (0.15 s); intermediate ones are dropped.
- Kasa: python-kasa discovery + query with the same 1 s / 2 retries budget, one try once known down.
  The TCP session is reused; a reset mid-session reconnects on the retry.
- Nothing is logged above WARNING, and credentials are never logged.

## 7. Dependencies (requirements/devices.txt) and aarch64

- `python-kasa==0.11.0.1`: pure Python. Its dependencies as resolved on 2026-10-24: `aiohttp 3.14.5`,
  `aiosignal`, `aiohappyeyeballs`, `asyncclick`, `attrs`, `frozenlist`, `idna`, `mashumaro`,
  `multidict`, `propcache`, `yarl` ship universal (`py3-none-any`) wheels; `cryptography 50.0.2` and
  `cffi 2.1.1` ship `manylinux aarch64` wheels for CPython 3.12. Nothing needs a compiler on the GB10.
- WiZ: no dependency (own client). `pywizlight` was the alternative; not used, to keep the retry /
  debounce / never-raise behaviour explicit and testable against `FakeWizBulb`.

## 8. Risk

The bulb takes unauthenticated commands from anyone on its Wi-Fi: keep it on our own router with a
secret WPA2 key, never on the venue network; if the lamp is lost the room still runs (unhealthy fake).
