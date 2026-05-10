"""
IRCS – Intelligent Room Conditioning System
Context-Aware Environmental Control for Elderly and Infant Care.

Physical wiring reference (BCM numbering):
  GPIO 2  (SDA1) Pin 3  → ADS1115 SDA, BMP280 SDA, OLED SDA  [I2C Data]
  GPIO 3  (SCL1) Pin 5  → ADS1115 SCL, BMP280 SCL, OLED SCL  [I2C Clock]
  GPIO 4         Pin 7  → DHT11 OUT                            [Digital IN]
  GPIO 17        Pin 11 → HC-SR04 TRIG                         [Digital OUT]
  GPIO 18        Pin 12 → MG90S Servo signal (orange)          [PWM]
  GPIO 21        Pin 40 → LED anode (via 330Ω)                 [Digital OUT]
  GPIO 23        Pin 16 → NPN Base (via 1kΩ) → Buzzer          [Digital OUT]
  GPIO 27        Pin 13 → HC-SR04 ECHO (via voltage divider)   [Digital IN]
"""

import os

# ── Hardware pin assignments (BCM numbering for Raspberry Pi) ─────────────────
ULTRASONIC_TRIG_PIN = 17       # HC-SR04 TRIG
ULTRASONIC_ECHO_PIN = 27       # HC-SR04 ECHO (voltage-divider protected)

DHT11_PIN = 4                  # DHT11 data pin

# ── I2C device addresses ──────────────────────────────────────────────────────
BMP280_I2C_ADDRESS  = 0x76     # 0x77 if SDO pulled high
ADS1115_I2C_ADDRESS = 0x48     # default ADDR pin → GND

# ADS1115 channel assignments (single-ended, A0-A3)
MQ135_ADC_CHANNEL = 0          # A0 – MQ-135 air-quality sensor
LDR_ADC_CHANNEL   = 1          # A1 – LDR

# ADS1115 gain setting (±4.096 V covers the 0-3.3 V sensor range)
ADS1115_GAIN = 1

# ── Camera settings ──────────────────────────────────────────────────────────
CAMERA_INDEX        = 0
CAMERA_FRAME_WIDTH  = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_FPS          = 15

# ── Actuator GPIO pins ───────────────────────────────────────────────────────
SERVO_PIN  = 18                # MG90S servo signal (hardware PWM)
LED_PIN    = 21                # LED anode via 330Ω resistor
BUZZER_PIN = 23                # NPN transistor base via 1kΩ → buzzer

# Servo PWM: 50 Hz carrier; duty cycle maps to ventilation damper position
SERVO_PWM_FREQ  = 50
SERVO_DUTY_STOP = 0.0          # idle / off
SERVO_DUTY_SLOW = 5.0          # ~90° – moderate ventilation
SERVO_DUTY_FULL = 12.0         # ~180° – full ventilation

# ── Sampling & timing ────────────────────────────────────────────────────────
SENSOR_POLL_INTERVAL   = 10    # seconds between sensor cycles
ROLLING_WINDOW_SECONDS = 30    # seconds of history for derived features
LLM_CALL_INTERVAL      = 300   # seconds between scheduled LLM calls (5 min)
ACTUATOR_RAMP_SECONDS  = 900   # 15-minute gradual transition between states

# ── Presence detection ────────────────────────────────────────────────────────
PRESENCE_DISTANCE_CM   = 400   # HC-SR04 returns > this → ROOM_EMPTY standby

# ── Context-state labels ──────────────────────────────────────────────────────
LABEL_MAP = {
    0: "ROOM_EMPTY",
    1: "ACTIVE_AWAKE",
    2: "RESTING",
    3: "SLEEPING",
}

# ── Evidence-based environmental targets per context state ────────────────────
# Format: {state: {"temp": °C, "humidity": %RH, "co2_max": ppm, "lux": lux}}
# Targets are for elderly occupants by default; infant targets are tighter.
COMFORT_TARGETS = {
    "ROOM_EMPTY":   {"temp": 18.0, "humidity": 50.0, "co2_max": 1000, "lux": 0},
    "ACTIVE_AWAKE": {"temp": 21.0, "humidity": 50.0, "co2_max": 900,  "lux": 300},
    "RESTING":      {"temp": 20.0, "humidity": 55.0, "co2_max": 800,  "lux": 80},
    "SLEEPING":     {"temp": 18.5, "humidity": 60.0, "co2_max": 700,  "lux": 5},
}

# ── Safety hard-limit overrides (always active, regardless of ML output) ──────
SAFETY_TEMP_MIN_ELDERLY  = 18.0   # °C – immediate corrective heat if below
SAFETY_TEMP_MIN_INFANT   = 20.0   # °C – tighter floor for infant mode
SAFETY_TEMP_MAX          = 30.0   # °C – immediate ventilation if above
SAFETY_CO2_MAX_PPM       = 1500   # ppm – immediate ventilation + alert
SAFETY_TEMP_DROP_RATE    = 2.0    # °C/min – rapid drop triggers override
SAFETY_CONFIDENCE_FLOOR  = 0.70   # predict_proba below this → fallback to CV

# ── Machine-learning ─────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "ml", "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "ml", "scaler.pkl")

# ── LLM settings ────────────────────────────────────────────────────────────
# Provider priority: Groq (free) → Anthropic (paid) → Pollinations (no-key)
# Set ONE of the API keys below; leave the others blank.
#
# Groq  – free, fast.  Sign up at https://console.groq.com  (no credit card)
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.3-70b-versatile"  # best free Groq model
#
# Anthropic – $5 free credit on signup at https://console.anthropic.com
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL         = "claude-3-5-haiku-20241022"
#
LLM_MAX_TOKENS    = 300
LLM_CALL_INTERVAL = 300   # seconds between LLM calls (5 min)

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "ircs.db")

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST  = "0.0.0.0"
DASHBOARD_PORT  = 5000
DASHBOARD_DEBUG = False
