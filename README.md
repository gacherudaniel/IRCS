# IRCS – Intelligent Room Conditioning System

**Context-aware environmental control for elderly and infant care, running on a Raspberry Pi 4B.**

IRCS continuously monitors a room using a sensor array, classifies the current occupancy context using a machine-learning model, drives actuators to maintain evidence-based comfort targets, and generates plain-English caregiver status updates via an LLM. A live web dashboard provides real-time visibility.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Hardware & Sensors](#hardware--sensors)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [LLM Explainer](#llm-explainer)
6. [Actuator Control](#actuator-control)
7. [Web Dashboard](#web-dashboard)
8. [Database](#database)
9. [Configuration](#configuration)
10. [Installation & Setup](#installation--setup)
11. [Running the System](#running-the-system)
12. [Project Structure](#project-structure)

---

## Overview

IRCS detects four room context states and adjusts the environment accordingly:

| State | Meaning |
|---|---|
| `ROOM_EMPTY` | No occupant detected — standby mode |
| `ACTIVE_AWAKE` | Occupant is moving and awake |
| `RESTING` | Occupant is still but awake (sitting/reclined) |
| `SLEEPING` | Occupant is horizontal and still |

For each state, evidence-based environmental targets are enforced:

| State | Temp (°C) | Humidity (%RH) | CO₂ max (ppm) | Lux |
|---|---|---|---|---|
| `ROOM_EMPTY` | 18.0 | 50 | 1000 | 0 |
| `ACTIVE_AWAKE` | 21.0 | 50 | 900 | 300 |
| `RESTING` | 20.0 | 55 | 800 | 80 |
| `SLEEPING` | 18.5 | 60 | 700 | 5 |

A safety override layer operates independently of the ML model and immediately triggers full ventilation and a buzzer alert if temperature or CO₂ exceed hard limits.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Raspberry Pi 4B                          │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐   │
│  │   Sensors   │──▶│  Feature     │──▶│  ML Classifier     │   │
│  │  (I2C/GPIO/ │   │  Extractor   │   │  (Random Forest)   │   │
│  │   Camera)   │   │  10 features │   │  4 context states  │   │
│  └─────────────┘   └──────────────┘   └────────┬───────────┘   │
│                                                │               │
│                          ┌─────────────────────┘               │
│                          ▼                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐   │
│  │  Actuator   │◀──│  Controller  │   │   LLM Explainer    │   │
│  │  (Servo /   │   │  + Safety    │   │  (Gemini/Groq/     │   │
│  │  LED/Buzzer)│   │  Overrides   │   │  Anthropic/local)  │   │
│  └─────────────┘   └──────────────┘   └────────┬───────────┘   │
│                                                │               │
│  ┌─────────────┐                    ┌──────────▼───────────┐   │
│  │  SQLite DB  │◀───────────────────│   Flask Dashboard    │   │
│  │  (WAL mode) │                    │   /  /api/status     │   │
│  └─────────────┘                    │   /api/history       │   │
│                                     └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Main Control Loop

The system runs a background **sensor loop thread** (`sensor_loop` in `main.py`) that repeats every `SENSOR_POLL_INTERVAL` seconds (default: 10 s):

1. **Read sensors** — camera, BMP280, MQ-135, LDR all polled in one cycle.
2. **Extract features** — 30-second rolling window produces a 10-element feature vector.
3. **Classify** — Random Forest predicts the context state; CV heuristic used as fallback if confidence < 70%.
4. **Apply actuators** — servo, LED, and buzzer adjusted toward comfort targets with a 15-minute gradual ramp.
5. **Log to database** — reading, label, and LLM explanation stored in SQLite.
6. **Call LLM** — natural-language caregiver update generated every 5 minutes or on state change.
7. **Update shared state** — Flask dashboard reads from this dict to serve live data.

---

## Hardware & Sensors

All sensors share the I2C bus (GPIO 2 SDA, GPIO 3 SCL) except the camera and ultrasonic sensor.

### BMP280 — Temperature & Pressure (`sensors/bmp280_sensor.py`)

- **Interface:** I2C at address `0x76`
- **Library:** Adafruit CircuitPython BMP280
- **Outputs:** temperature (°C), barometric pressure (hPa), altitude (m)
- Temperature is the primary thermal input for the ML model and comfort control.

### MQ-135 Air Quality Sensor (`sensors/air_quality.py`)

- **Interface:** Analogue → ADS1115 channel A0
- **Principle:** Measures combustible/harmful gas concentration; used as a CO₂ proxy indoors.
- **Outputs:** raw ADC counts → scaled ppm equivalent (0–2000 ppm), qualitative label (`good` / `moderate` / `poor` / `hazardous`)
- CO₂ estimate is used both in the feature vector and in safety hard-limit checks.

### ADS1115 ADC (`sensors/adc.py`)

- **Interface:** I2C at address `0x48`, gain = ±4.096 V (16-bit resolution)
- Acts as the ADC bridge for both the MQ-135 (A0) and LDR (A1).
- Channels A2 and A3 are spare.

### LDR — Light Level (`sensors/ldr.py`)

- **Interface:** Analogue voltage divider → ADS1115 channel A1
- **Output:** lux estimate using a logarithmic curve fit (~1–100 000 lux range)
- Used for both ML classification (room lit vs. dark) and LED control.

### Camera — Posture & Motion (`sensors/camera.py`)

- **Interface:** USB/CSI camera via OpenCV (`cv2.VideoCapture`)
- **Resolution:** 640×480 @ 15 fps (configurable)
- Produces two outputs:
  - **Posture** — MediaPipe Pose Landmarker compares average shoulder and hip y-coordinates (normalised 0–1) to classify `UPRIGHT` (0), `RECLINED` (1), or `HORIZONTAL` (2).
  - **Flow score** — Farneback dense optical flow computes frame-to-frame motion magnitude, normalised to 0.0 (still) – 1.0 (high motion).
- The camera pipeline is designed to be gated by the ultrasonic sensor confirming occupancy.

### HC-SR04 — Ultrasonic Distance (`sensors/ultrasonic.py`)

- **Interface:** GPIO (TRIG out, ECHO in)
- **Principle:** Measures round-trip echo time; distance = (pulse duration × 34 300 cm/s) / 2
- Used as a presence gate: the camera is only activated when an occupant is detected within range.
- Includes a 40 ms software timeout to prevent blocking on missed echoes.

---

## Machine Learning Pipeline

### Feature Extraction (`ml/feature_extractor.py`)

A 30-second rolling window of readings is maintained. On each call to `extract()`, a **10-element feature vector** is assembled:

| Index | Feature | Source |
|---|---|---|
| 0 | `temperature` | BMP280 °C |
| 1 | `pressure` | BMP280 hPa |
| 2 | `co2_ppm` | MQ-135 ppm estimate |
| 3 | `lux` | LDR lux |
| 4 | `flow_score` | Farneback optical flow (0–1) |
| 5 | `lux_rate` | Rate of lux change over rolling window (lux/s) |
| 6 | `hour_sin` | Cyclic sin encoding of hour-of-day |
| 7 | `hour_cos` | Cyclic cos encoding of hour-of-day |
| 8 | `dow_sin` | Cyclic sin encoding of day-of-week |
| 9 | `dow_cos` | Cyclic cos encoding of day-of-week |

Circadian features (hour/day encodings) give the model temporal awareness — e.g., distinguishing daytime rest from nighttime sleep. The vector is optionally scaled by a pre-fitted `StandardScaler` (loaded from `ml/scaler.pkl`).

### Classifier (`ml/classifier.py`)

- **Model:** Scikit-learn Random Forest loaded from `ml/model.pkl` via `joblib`.
- **Output:** `(class_id, confidence)` where class_id maps to one of the four context states.
- **Confidence gate:** if `predict_proba()` returns a max confidence below `SAFETY_CONFIDENCE_FLOOR` (default: 0.70), the classifier falls back to a rule-based CV heuristic:

```
dark + still → SLEEPING
high motion  → ACTIVE_AWAKE
present + dim → RESTING
very dark + low CO₂ → ROOM_EMPTY
```

### Training (`ml/train.py`)

Training data is collected via `data/collect.py` (live sensor logs) or generated synthetically with `data/generate_synthetic_data.py`. The `data/preprocess.py` script cleans and engineers features before model fitting.

---

## LLM Explainer

**File:** `llm/explainer.py`

The `LLMExplainer` generates a 2–3 sentence plain-English caregiver status update from the current sensor readings and classified context state. It is called:
- Every `LLM_CALL_INTERVAL` seconds (default: **5 minutes**).
- Immediately whenever the predicted context state changes.

### Provider Priority

The explainer tries providers in this order, using the first one with a valid API key:

1. **Google Gemini** (`gemini-2.0-flash`) — generous free tier (1 500 req/day). Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. **Groq** (`llama-3.3-70b-versatile`) — free tier, very fast inference. Sign up at [console.groq.com](https://console.groq.com).
3. **Anthropic** (`claude-3-5-haiku`) — $5 free credit on signup.
4. **Pollinations.ai** — zero signup, completely free OpenAI-compatible public endpoint.
5. **Local fallback templates** — always works offline; hard-coded messages per context state.

### How the Prompt is Built

The `_build_prompt()` function assembles a structured message containing:
- Current timestamp
- Detected context state and ML confidence percentage
- All sensor readings: temperature, humidity, CO₂, lux, distance, posture code, motion score

This is sent alongside a system prompt instructing the model to act as a calm, professional room-monitoring assistant for elderly/infant care.

### System Prompt

> *"You are an intelligent room-monitoring assistant for a care system looking after elderly people and infants. Given current sensor readings and the detected context state, produce a brief (2-3 sentence), plain-English status update for caregivers. Highlight any comfort or safety concerns and suggest one actionable step if needed. Use a calm, professional tone."*

### Setup (Gemini — recommended)

```bash
# Add to ircs/.env
GEMINI_API_KEY=AIza...
```

---

## Actuator Control

**File:** `actuators/controller.py`

Three actuators are driven via Raspberry Pi GPIO (BCM numbering):

| Actuator | GPIO | Purpose |
|---|---|---|
| MG90S Servo | GPIO 18 (PWM) | Ventilation damper — controls airflow |
| LED | GPIO 21 | Room illumination |
| Buzzer | GPIO 23 (NPN) | Audible caregiver safety alert |

### Gradual Ramps

When the context state changes, the servo duty cycle moves incrementally toward the new target over `ACTUATOR_RAMP_SECONDS` (default: **15 minutes**). This prevents abrupt airflow changes from disturbing a sleeping occupant.

### Safety Override Layer

Always active regardless of ML output. Hard limits trigger immediately:

| Condition | Action |
|---|---|
| Temperature < 18 °C (elderly) / 20 °C (infant) | Full heating / ventilation + buzzer |
| Temperature > 30 °C | Full ventilation + buzzer |
| CO₂ > 1 500 ppm | Full ventilation + buzzer |
| Rapid temperature drop (> 2 °C/min) | Immediate corrective response + buzzer |

---

## Web Dashboard

**File:** `dashboard/app.py`

A Flask web application serves a live monitoring UI:

| Route | Description |
|---|---|
| `GET /` | Main dashboard page (HTML) |
| `GET /api/status` | Latest sensor reading, room state, and LLM explanation (JSON) |
| `GET /api/history?n=50` | Last N log rows from SQLite (JSON, clamped 1–500) |

The dashboard reads from a **shared state dict** updated by the sensor loop thread, making it fully non-blocking. Access it at `http://<pi-ip>:5000` by default.

---

## Database

**File:** `database/logger.py`

All sensor readings and ML predictions are persisted to **SQLite** (`data/ircs.db`) in WAL journal mode for concurrent read/write safety.

Schema:

```sql
CREATE TABLE sensor_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT,
    temperature  REAL,
    pressure     REAL,
    altitude     REAL,
    air_quality  INTEGER,
    ldr          REAL,
    flow_score   REAL,
    label        TEXT,
    explanation  TEXT
);
```

---

## Configuration

All tunable parameters live in `ircs/config.py`. Key settings:

```python
SENSOR_POLL_INTERVAL   = 10    # seconds between sensor cycles
ROLLING_WINDOW_SECONDS = 30    # rolling window for derived features
LLM_CALL_INTERVAL      = 300   # seconds between scheduled LLM calls
ACTUATOR_RAMP_SECONDS  = 900   # 15-minute gradual actuator transition

SAFETY_TEMP_MIN_ELDERLY = 18.0  # °C
SAFETY_TEMP_MAX         = 30.0  # °C
SAFETY_CO2_MAX_PPM      = 1500  # ppm
SAFETY_CONFIDENCE_FLOOR = 0.70  # ML confidence threshold
```

LLM API keys are read from environment variables (recommended: use an `.env` file):

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...
```

---

## Installation & Setup

**Requirements:** Raspberry Pi 4B, Python 3.11+, Raspberry Pi OS (64-bit recommended).

```bash
cd ircs
bash setup.sh
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GEMINI_API_KEY
```

To train a fresh ML model from collected data:

```bash
python data/collect.py        # collect live training data
python data/preprocess.py     # clean and engineer features
python ml/train.py            # train Random Forest, saves model.pkl + scaler.pkl
```

---

## Running the System

```bash
cd ircs
source venv/bin/activate
python main.py
```

The system:
1. Initialises all sensors, ML pipeline, actuators, database, LLM, and dashboard.
2. Starts the sensor loop in a background daemon thread.
3. Serves the Flask dashboard on `http://0.0.0.0:5000`.
4. Shuts down cleanly on `Ctrl+C`, releasing GPIO and camera resources.

To run tests without hardware:

```bash
python test_sensors.py
```

All sensors and actuators fall back to simulation mode (random values) when hardware libraries (`RPi.GPIO`, `adafruit_bmp280`, `cv2`, `mediapipe`) are not available.

---

## Project Structure

```
ircs/
├── main.py                    # Entry point — sensor loop + dashboard
├── config.py                  # All configuration constants and API keys
├── requirements.txt
├── setup.sh
├── test_sensors.py
├── sensors/
│   ├── adc.py                 # ADS1115 16-bit I2C ADC driver
│   ├── air_quality.py         # MQ-135 CO₂/air quality (via ADS1115)
│   ├── bmp280_sensor.py       # BMP280 temperature + pressure (I2C)
│   ├── camera.py              # MediaPipe posture + optical flow motion
│   ├── ldr.py                 # LDR light level (via ADS1115)
│   └── ultrasonic.py          # HC-SR04 presence/distance
├── ml/
│   ├── feature_extractor.py   # 10-feature rolling-window vector builder
│   ├── classifier.py          # Random Forest wrapper + CV fallback
│   └── train.py               # Model training script
├── llm/
│   └── explainer.py           # Multi-provider LLM caregiver updates
├── actuators/
│   └── controller.py          # Servo / LED / buzzer + safety overrides
├── dashboard/
│   ├── app.py                 # Flask routes + API endpoints
│   └── templates/index.html   # Live monitoring UI
├── database/
│   └── logger.py              # SQLite WAL logger
└── data/
    ├── collect.py             # Live data collection
    ├── generate_synthetic_data.py
    ├── preprocess.py
    └── populate_db.py
```
