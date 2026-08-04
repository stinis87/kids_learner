---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini conversation app

Conversational app for the Reachy Mini robot combining realtime voice backends and choreographed motion libraries.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Advanced features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Overview
- Real-time audio conversation loop for low-latency streaming, powered by the **Hugging Face** realtime backend using the built-in Hugging Face server or your own local endpoint.
- Vision is handled by the realtime backend when the `camera` tool is used.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble.
- Async tool dispatch integrates robot motion and camera capture. An optional web UI (`--ui`) provides personality selection, mic control, and settings.

## Architecture

The app follows a layered architecture connecting the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).<br>
> Windows support is currently experimental and has not been extensively tested. Use with caution.

<details open>
<summary><b>Using uv (recommended)</b></summary>

Set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

> **Note:** To reproduce the exact dependency set from this repo's `uv.lock`, run `uv sync --frozen`. This ensures `uv` installs directly from the lockfile without re-resolving or updating any versions.

Include dev dependencies:
```bash
uv sync --group dev
```

</details>

<details>
<summary><b>Using pip</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Install dev dependencies:**
```bash
pip install -e .[dev]                   # Development tools
```

</details>

### Dependency groups

| Group | Purpose | Notes |
|-------|---------|-------|
| `dev` | Developer tooling (`pytest`, `ruff`, `mypy`) | Development-only dependencies. Use `--group dev` with uv or `[dev]` with pip. |

## Configuration

The default setup uses the Hugging Face backend and does not require an API key.

Copy `.env.example` to `.env` when you want to point Hugging Face at your own local endpoint.

| Variable | Description |
|----------|-------------|
| `REALTIME_TRANSCRIPTION_LANGUAGE` | Optional input transcription language for the realtime backend. Defaults to `en`; set to a backend-supported code such as `zh` for Chinese. |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face connection selector: `deployed` uses the built-in Hugging Face server; `local` uses `HF_REALTIME_WS_URL`. Defaults to `deployed`. |
| `HF_REALTIME_WS_URL` | Direct websocket endpoint for your own Hugging Face backend. Accepts either a base URL like `ws://127.0.0.1:8765/v1` or the full websocket URL `ws://127.0.0.1:8765/v1/realtime`. Used when `HF_REALTIME_CONNECTION_MODE=local`. |
| `HF_TOKEN` | Optional token for Hugging Face access (for gated/private assets). |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | Minutes of inactivity before Reachy goes to sleep and the app stops. Defaults to `1440` (one day); set to `0` to disable. |
| `WAKE_WORD_ENABLED` | Gate the microphone behind a wake word. Defaults to `1` (enabled). |
| `WAKE_WORD_MODEL` | Pretrained [openWakeWord](https://github.com/dscripka/openWakeWord) model to listen for. Defaults to `alexa`. |
| `WAKE_WORD_THRESHOLD` | Detection score (0-1) required to trigger the wake word. Defaults to `0.5`. |
| `KID_VOICE_EFFECT_ENABLED` | Pitch the assistant's TTS voice up and add a light robotic ring-modulation touch. Defaults to `1` (enabled). |
| `KID_VOICE_PITCH_FACTOR` | Pitch/speed multiplier applied to the voice (`>1` raises pitch). Defaults to `1.25`. |
| `KID_VOICE_ROBOT_MIX` | Ring-modulation wet mix (0-1). Defaults to `0.2`. |
| `KID_VOICE_ROBOT_CARRIER_HZ` | Ring-modulation carrier frequency in Hz. Defaults to `60`. |

### Hugging Face Connection Modes

Use the built-in Hugging Face server through the app-managed Space proxy. This is the default for a new install; set it explicitly only when you want to switch back from a saved local endpoint:

```env
HF_REALTIME_CONNECTION_MODE=deployed
```

Run your own realtime voice backend using [speech-to-speech](https://github.com/huggingface/speech-to-speech) on the same machine as the conversation app:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

Run your own Hugging Face backend on your laptop and connect to it from Reachy Mini Wireless over the same Wi-Fi network:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

For that LAN setup, make sure the backend listens on an address reachable from the robot, not only on `127.0.0.1`.

If the backend stays bound to loopback on your laptop, you can forward it into the robot over SSH instead:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

Then set this on the robot:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

In the web UI's Settings view, the Connection section lets you choose either the built-in server or a local `host:port` target. The UI writes `HF_REALTIME_CONNECTION_MODE` for you, and the local path writes `HF_REALTIME_WS_URL` with a default of `localhost:8765`.

## Running the app

Activate your virtual environment, then launch:

```bash
reachy-mini-conversation-app
```

> [!TIP]
> Make sure the Reachy Mini daemon is running before launching the app. If you see a `TimeoutError`, it means the daemon isn't started. See [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) for setup instructions.

The app runs in console mode by default. Add `--ui` to also serve a web UI at http://127.0.0.1:7860/ for picking a personality, controlling the mic, and changing settings. All options are described in the CLI table below.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-camera` | `False` | Run without camera capture. |
| `--ui` | `False` | Serve the web UI at http://127.0.0.1:7860/, in addition to console mode. |
| `--robot-name` | `None` | Optional. Connect to a specific robot by name when running multiple daemons on the same subnet. See [Multiple robots on the same subnet](#advanced-features). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |
| `--ring-login` | `False` | One-time interactive login to your Ring account for the `check_ring_camera` tool; caches a token and exits. See [Ring cameras](#advanced-features). |
| `--ring-check` | `False` | Verify the cached Ring login: lists your devices, fetches one snapshot from each, saves them as JPEGs in a local `ring_images/` folder (gitignored), then exits. See [Ring cameras](#advanced-features). |

### Examples

```bash
# Audio-only conversation (no camera)
reachy-mini-conversation-app --no-camera

# Launch with the minimal web UI for personality/mic/settings control
reachy-mini-conversation-app --ui
```

## LLM tools exposed to the assistant

The default profile exposes these tools. Custom profiles can enable a different set in their own `tools.txt`.

| Tool | Action | Dependencies |
|------|--------|--------------|
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face datasets. | Core install only. Uses the default open emotions dataset: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library). |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `camera` | Capture the latest camera frame and analyze it with the selected realtime backend. | Core install only. Requires the camera (disable with `--no-camera`). |
| `idle_do_nothing` | Explicitly remain idle during an idle turn. Not intended for normal conversation turns. | Core install only. |
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `head_tracking` | Follow the user's face with the head, or stop following. | Core install only. Requires a daemon with the `vision` extra and a camera. |
| `red_light_green_light` | Play one full round of Red Light, Green Light per call: spins its body around to turn its back on the user and hides its eyes, waiting a randomized moment (occasionally faking out with a lightning-fast one), then spins back to face the user, snaps its head up, watches for movement, and turns to look at whoever gets caught. | Core install only. Requires the camera (disable with `--no-camera`). Detects movement by diffing consecutive camera frames and turns toward wherever the change is concentrated. |
| `go_to_sleep` | Run Reachy's sleep movement and stop the current app after an explicit user request. | Core install only. |
| `sweep_look` | Sweep Reachy's head left, right, and back to center. | Bundled default profile tool. |
| `remember` | Save one short, stable fact about the user for future sessions. | Core install only. Stored in the app instance data directory. |
| `forget` | Remove a saved memory fact by matching a short query. | Core install only. |
| `pollen_robotics_reachy_mini_search_tool__search_web` | Search the web and return a short list of results. | Preinstalled MCP Space: `pollen-robotics/reachy-mini-search-tool`. |
| `pollen_robotics_reachy_mini_weather_tool__get_weather` | Report today's weather for a place: current conditions, high and low temperature, and rain chance. | Preinstalled MCP Space: `pollen-robotics/reachy-mini-weather-tool`. |
| `pollen_robotics_reachy_mini_time_tool__get_time` | Report the current time for a timezone or the user's local time, or the difference between two timezones. | Preinstalled MCP Space: `pollen-robotics/reachy-mini-time-tool`. |
| `switch_profile` | Switch Reachy's active personality profile by voice (e.g. "let's read a bedtime story", "go back to normal"). | Core install only. Restarts the realtime session, same as switching profiles from the UI. |
| `check_ring_camera` | Fetch a fresh snapshot from one Ring camera (by the name configured in the Ring app) or all of them at once, and describe what's happening. | Core install only. Requires a one-time `--ring-login`. See [Ring cameras](#advanced-features). |
| `check_ring_history` | Answer retroactive questions about a Ring camera's motion/doorbell events for a given day ("today", "yesterday", "day before yesterday", or a `YYYY-MM-DD` date), and optionally describe what the most recent one looked like. | Core install only. Requires a one-time `--ring-login`. Describing an event additionally requires `ffmpeg` and an active Ring Protect subscription. See [Ring cameras](#advanced-features). |
| `talk_to_door` | Open a live two-way audio call with a Ring doorbell/camera, so Reachy's speech is also spoken through that device's speaker and it can hear whoever is near it. | Core install only. Requires a one-time `--ring-login`. See [Ring doorbell calls](#advanced-features). |
| `end_door_call` | Hang up the active live doorbell call. | Core install only. See [Ring doorbell calls](#advanced-features). |

> [!NOTE]
> `remember`/`forget` facts are stored in `memory.v1.json` inside the app's instance data directory (`~/.local/share/reachy_mini_conversation_app/` by default, or the instance path used by the desktop launcher). `forget` only removes facts matched by query. To reset all remembered facts, delete this file.

## Advanced features

Built-in motion content is published as open Hugging Face datasets:
- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>Wake word</b></summary>

Reachy starts awake and listening. Saying the wake word (default: "Alexa") puts it to sleep — no audio reaches the conversation backend while dormant — and saying it again wakes it back up. Detection runs fully on-device using a pretrained [openWakeWord](https://github.com/dscripka/openWakeWord) model, so no extra setup or network access is required beyond the one-time model download on first run.

Reachy tucks its head down and relaxes its antennas while dormant, and snaps back up alert once woken up. If the model can't be loaded (e.g. no network on first run), the app falls back to always listening rather than staying silent.

Set `WAKE_WORD_MODEL` to any other [pretrained openWakeWord model name](https://github.com/dscripka/openWakeWord#pretrained-models) (e.g. `hey_jarvis`, `hey_mycroft`), tune sensitivity with `WAKE_WORD_THRESHOLD`, or set `WAKE_WORD_ENABLED=0` to disable the gate entirely.

</details>

<details>
<summary><b>Ring cameras</b></summary>

The `check_ring_camera` tool lets Reachy answer "what's happening in the garden/at the front door/in the shed (bod)?" (or "check all cameras") by fetching a fresh snapshot from your Ring devices. It uses the unofficial [`ring_doorbell`](https://github.com/python-ring-doorbell/python-ring-doorbell) library against your own Ring account — Ring has no official public API for personal integrations, so this can break if Ring changes its backend.

In the Ring app, name each doorbell/camera the way you want to refer to it out loud — the tool matches your account's device names exactly (e.g. "Garden", "Front Door", "bod"). A few Norwegian synonyms also resolve to matching device names: "hage"/"hagen" → "Garden", "framsiden" → "Front Door".

Before first use, run the one-time interactive login, which caches an OAuth token (no password stored):

```bash
reachy-mini-conversation-app --ring-login
```

Then verify it actually works end to end — this reuses the same code path as the live tool, so a successful run means `check_ring_camera` will work in conversation:

```bash
reachy-mini-conversation-app --ring-check
```

This lists your Ring devices, fetches one snapshot from each, and saves them as `ring_images/ring_snapshot_<device>.jpg` (gitignored) so you can open them and confirm the image actually shows the right camera feed.

The token is a single Ring account credential shared by the whole app (not per-instance data), stored by default under your user data directory (`~/.local/share/reachy_mini_conversation_app/` on Linux/macOS). Override the location with `RING_TOKEN_CACHE_PATH`. Re-run `--ring-login` if the cached token expires.

> [!NOTE]
> The cached token file grants access to your Ring account, so it's written with owner-only permissions (`0600`) and, like `.env`, must never be committed or shared. During normal conversation, `check_ring_camera` sends each snapshot straight to the configured realtime backend (e.g. Hugging Face) for image analysis without ever writing it to disk — the same way the built-in `camera` tool already sends webcam frames. `--ring-check` is the only place snapshots are saved locally (into `ring_images/`, already gitignored), purely so you can visually confirm the feed is correct; delete that folder once you're done.

**Proactively reacting to motion/doorbell events**: beyond answering when asked, Reachy can also notice events on its own. This is enabled by default for every profile whenever Ring is configured — a profile can tune the timing, or opt out entirely with `enabled=false`, via an optional `ring_watcher.txt` file in its profile directory (see `profiles/default/ring_watcher.txt`). When enabled, Reachy polls each Ring device's event history (the same "motion"/"ding" events shown in the Ring app) every `poll_interval_seconds`, and reacts to a genuinely new event with a physical `play_emotion` reaction ("excited" for a doorbell ring, "attentive" for motion) plus fetching a fresh snapshot and prompting itself to look and comment — at most once per device every `device_cooldown_seconds`, to avoid nagging about repeated motion (e.g. wind, passing cars). Ring has no realtime push API usable here (that would require registering as a Firebase Cloud Messaging mobile client, which `ring_doorbell` doesn't support), so this is polling-based; like `check_ring_camera`, the reaction snapshot is only ever sent to the realtime backend for analysis and never written to disk.

**Asking about the past**: `check_ring_history` answers questions like "was anyone at the front door today?", "how many times did the doorbell ring yesterday?", or "what happened in the garden on 2024-01-05?" — day names ("today"/"yesterday"/"day before yesterday") and locations both also understand Norwegian ("i dag"/"i går"/"i forgårs", "hage", "framsiden"), the same way `check_ring_camera` already does for locations. It pages through the device's Ring event history and, for a broad question, reports back only counts and timestamps — fast, and requires nothing beyond the same login as `check_ring_camera`. When the question is about a specific event — the most recent one, an ordinal like "the second one" or "the earliest", or an approximate clock time like "the 2pm one" ("andre", "første", "kl. 14" in Norwegian) — it additionally downloads that event's recorded clip and extracts a handful of stills spread across it for Reachy to describe, since the realtime backend only ever looks at still images, never video — Reachy is instructed to say it's checking and might take a moment before this slower step, rather than going quiet. That step needs:
- an active **Ring Protect subscription** (Ring only lets recordings be downloaded with one — without it, Reachy will say so instead of failing silently), and
- **`ffmpeg`** installed and on `PATH` (e.g. `brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu, or the [official builds](https://ffmpeg.org/download.html) on Windows) — used to pull frames out of the downloaded clip. Nothing is written to disk beyond a temporary file deleted immediately after extraction.

</details>

<details>
<summary><b>Ring doorbell calls</b></summary>

Beyond snapshots, Reachy can open a live two-way audio call with a Ring doorbell or camera and talk through it, using the same personality, voice, and tools as the rest of the conversation. This uses `ring_doorbell`'s experimental WebRTC live-view signaling together with [`aiortc`](https://github.com/aiortc/aiortc) for the actual audio, against a device's own speaker and microphone — it requires a model that has both (most video doorbells/battery cams qualify; a camera-only device would only support hearing, not speaking).

Two ways to start a call:
- **On request, any time** — say something like "let me talk to the front door" or "tell the person outside I'm coming". Reachy calls the `talk_to_door` tool and, from then on, its replies are spoken both in the room and through the device's speaker, and it can hear whoever is near it.
- **On a doorbell ring** — controlled per profile by `door_call_mode` in `ring_watcher.txt` (see [`profiles/default/ring_watcher.txt`](profiles/default/ring_watcher.txt)):
  - `ask` (default) — a "ding" has Reachy ask out loud whether it should answer the door itself or let you talk, then opens the call accordingly based on your reply.
  - `auto` — opens the call immediately with no question asked, so Reachy answers fully on its own.
  - `off` — back to the original snapshot-only nudge, no call ever offered.

  Motion events are unaffected by `door_call_mode` and always just get the usual snapshot nudge.

Either way, say goodbye or ask to hang up and Reachy calls `end_door_call` to close the call; it also closes automatically after 5 minutes as a safety cutoff, or if the visitor's end disconnects first. Requires the same one-time `--ring-login` as `check_ring_camera`.


</details>


<details>
<summary><b>Custom profiles</b></summary>

Create custom profiles with dedicated instructions and enabled tools.

For normal usage, select a profile from the UI and save it for startup. That selection is persisted in `startup_settings.json`.

If no startup settings have been saved yet, you can still seed startup from the environment with `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `profiles/<name>/`. If neither is set, the `default` profile is used.

Each profile should include `instructions.txt` (prompt text). If that file is missing or empty, the app logs a warning and falls back to `profiles/default/instructions.txt`. `greeting.txt` is optional and controls how the robot should start the conversation after the backend connects. `tools.txt` (list of allowed tools) is recommended. If missing for a non-default profile, the app falls back to `profiles/default/tools.txt`. Profiles can optionally contain custom tool implementations.

**Startup greeting:**

On startup, once the realtime backend is connected and ready, the app sends the active profile's `greeting.txt` as an internal text turn so the model opens with a fresh spoken greeting. Keep this file as a short instruction, not a fixed script, for example:
```
Greet me warmly in one sentence, in character, and vary the wording each time.
```
If `greeting.txt` is missing, the app uses the built-in default greeting prompt.

**Enabling tools:**

List enabled tools in `tools.txt`, one per line. Prefix with `#` to comment out:
```
play_emotion
# move_head

# My custom tool defined locally
sweep_look
```
Tools are resolved first from Python files in the profile folder (custom tools), then from the core library `src/reachy_mini_conversation_app/tools/` (like `dance`, `camera`).
Installed Hugging Face Space tools can also be enabled here after you add them with `tool-spaces`.

**Custom tools:**

On top of built-in tools found in the core library, you can implement custom tools specific to your profile by adding Python files in the profile folder.
Custom tools must subclass `reachy_mini_conversation_app.tools.core_tools.Tool` (see that module for the interface).

**Edit personalities from the UI:**

When running with `--ui`, the Home view lists available profiles (folders under `profiles/`) plus the built-in default:
- Tap a card to apply that personality and start talking.
- Tap "Custom" to create a new personality by entering a name, instructions, and an optional startup greeting prompt. It copies `tools.txt` from the `default` profile and stores the files under `user_personalities/<name>/` in the app instance directory (next to `.env`/`startup_settings.json`).

Note: switching a personality reloads its instructions and tools in place via a quick backend reconnect — no app restart. Editing the active profile's files on disk needs a re-select (or restart) to apply.

**Switching profiles by voice:**

Every profile enables the `switch_profile` tool, so the assistant can change personality mid-conversation when the user asks for a different companion (e.g. "let's read a bedtime story", "switch to night story reader", "go back to normal") — no need to touch the UI. It reuses the same apply-personality path as the UI, including the brief session restart.

</details>

<details>
<summary><b>Night story reader profile</b></summary>

`profiles/night_story_reader/` is a bedtime-story companion: it watches the picture book continuously and speaks up on its own — commenting on the page or asking an engaging question — instead of waiting to be asked "what do you see". This is driven by the **proactive vision engine** (`src/reachy_mini_conversation_app/proactive_vision.py`), which samples the camera every few seconds but only prompts the model for a spoken reaction when either the page/scene visibly changed or a short quiet pause has elapsed, whichever comes first — so it feels attentive without narrating non-stop.

Any profile can opt into this behavior by adding a `proactive_vision.txt` marker file to its profile folder, with optional overrides:
```
sample_interval_seconds=3
quiet_pause_seconds=7
speak_cooldown_seconds=10
```

</details>

<details>
<summary><b>Speaker-facing (mic array + face tracking)</b></summary>

The `default` profile additionally turns toward whoever is actually speaking by combining the ReSpeaker mic array's Direction-of-Arrival (DoA) with the existing camera-based face tracking (`head_tracking` tool): while head tracking is on and no face is currently locked, Reachy nudges its body yaw toward the detected speech direction, then hands back off to face tracking once it acquires a face there. If a different person then starts speaking from a meaningfully different direction (e.g. someone behind the robot) and that keeps up for about 1.5 seconds, Reachy redirects toward them instead of staying stuck on whoever it locked onto first — brief noises or echoes from another direction aren't enough to yank attention away. This requires ReSpeaker firmware >= 2.1.0; it degrades gracefully (no-op) on units without the mic array.

Enable this for a profile by adding an (empty) `speaker_tracking.txt` marker file to its profile folder — it's on by default for `profiles/default/` and intentionally left off for `night_story_reader` so the robot stays focused on the book instead of turning toward background noise.

</details>

<details>
<summary><b>Locked profile mode</b></summary>

To create a locked variant of the app that cannot switch profiles, edit `src/reachy_mini_conversation_app/config.py` and set the `LOCKED_PROFILE` constant to the desired profile name:
```python
LOCKED_PROFILE: str | None = "mars_rover"  # Lock to this profile
```
When `LOCKED_PROFILE` is set, the app always uses that profile, ignoring saved startup settings, `REACHY_MINI_CUSTOM_PROFILE`, and the web UI. The UI shows "(locked)" and disables all profile editing controls.
This is useful for creating dedicated clones of the app with a fixed personality. Clone scripts can simply edit this constant to lock the variant.

</details>

<details>
<summary><b>External profiles and tools</b></summary>

You can extend the app with profiles/tools stored outside the repository defaults.

- Core profiles are under `profiles/`.
- Core tools are under `src/reachy_mini_conversation_app/tools/`.

**Recommended layout:**

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       ├── instructions.txt
│       ├── greeting.txt     # optional startup greeting prompt
│       ├── tools.txt        # optional (see fallback behavior below)
│       └── voice.txt        # optional
├── external_tools/
│   └── my_custom_tool.py
└── installed_tool_spaces.json
```

**Environment variables:**

Set these values in your `.env` when you want env-driven external profile/tool selection:

```env
# Optional fallback/manual profile selector:
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
# Optional convenience mode:
# AUTOLOAD_EXTERNAL_TOOLS=1
```

**Loading behavior:**

- **Default/strict mode**: `tools.txt` defines enabled tools explicitly. Every name in `tools.txt` must resolve to either a built-in tool (`src/reachy_mini_conversation_app/tools/`) or an external tool module in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- **Convenience mode** (`AUTOLOAD_EXTERNAL_TOOLS=1`): all valid `*.py` tool files in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` are auto-added.
- **External profile fallback**: if the selected external profile has no `tools.txt`, the app falls back to built-in `profiles/default/tools.txt`.
- **Duplicate safety**: every loaded tool class must expose a unique `Tool.name`. The app now fails fast if two tool implementations claim the same tool name.

This supports both:
1. Local external tools used with built-in/default profile.
2. Local external profiles used with built-in default tools.

</details>

<details>
<summary><b>Hugging Face Space tools</b></summary>

You can install MCP-compatible Hugging Face Spaces as remote tool sources for this app. Private Spaces work too, as long as `HF_TOKEN` is set (or you have run `hf auth login`) for an account that can access them.

```bash
# install + enable in active profile
reachy-mini-conversation-app tool-spaces add <owner/space-name>

# enable in a specific profile
reachy-mini-conversation-app tool-spaces add <owner/space-name> --profile NAME

# install without enabling
reachy-mini-conversation-app tool-spaces add <owner/space-name> --install-only

# list installed spaces
reachy-mini-conversation-app tool-spaces list

# remove an installed space
reachy-mini-conversation-app tool-spaces remove owner/space-name
```

The bundled Pollen Spaces are enabled by default and resolve from static specs, so startup needs no Hugging Face discovery. For custom Spaces, the app validates the slug through the Hugging Face Hub, probes the standard MCP endpoint (sending the HF token only to private Spaces), discovers tools, enables them in the active profile's `tools.txt`, and writes the installed Space to:

- `installed_tool_spaces.json` in the managed app instance directory
- `external_content/installed_tool_spaces.json` in terminal mode

Recommended tags for discoverability on Hugging Face:

- `reachy-mini-tool`
- `mcp`

These tags are advisory only. Installation still relies on successful MCP validation, not on tag presence.

> [!NOTE]
> Preinstalled Pollen Spaces can be removed like any other (`tool-spaces remove pollen-robotics/reachy-mini-weather-tool`) or delete `installed_tool_spaces.json` to restore all defaults.

</details>

<details>
<summary><b>Multiple robots on the same subnet</b></summary>

If you run multiple Reachy Mini daemons on the same network, use:

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value so the app connects to the correct robot.

</details>

## Contributing

We welcome bug fixes, features, profiles, and documentation improvements. Please review our
[contribution guide](CONTRIBUTING.md) for branch conventions, quality checks, and PR workflow.
Working with an AI coding assistant? Point it at [`AGENTS.md`](AGENTS.md) — it codifies our engineering standards for agents.

Quick start:
- Fork and clone the repo
- Follow the [installation steps](#installation) (include the `dev` dependency group)
- Run contributor checks listed in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0
