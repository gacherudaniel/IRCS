"""
IRCS – Intelligent Room Control System
Global configuration settings.

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

# Servo PWM: 50 Hz carrier, duty 2-12 % → 0°-180°
SERVO_PWM_FREQ      = 50
SERVO_DUTY_STOP     = 0.0      # servo off / idle
SERVO_DUTY_SLOW     = 5.0      # ~90° (moderate ventilation)
SERVO_DUTY_FULL     = 12.0     # ~180° (full ventilation)


# ── Sampling intervals ────────────────────────────────────────────────────────
SENSOR_POLL_INTERVAL = 5        # seconds between sensor readings
CAMERA_POLL_INTERVAL = 1        # seconds between camera frames

# ── Thresholds ────────────────────────────────────────────────────────────────
TEMP_HIGH_THRESHOLD   = 28.0    # °C – trigger servo ventilation
TEMP_LOW_THRESHOLD    = 18.0    # °C – trigger buzzer alert
HUMIDITY_HIGH         = 70.0    # % RH – trigger ventilation
CO2_HIGH_THRESHOLD    = 1000    # ppm equivalent; trigger buzzer alert
LDR_DARK_THRESHOLD    = 10000   # lux; below = dark → turn on LED
DISTANCE_OCCUPIED_CM  = 200     # cm; below = room occupied

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "ircs.db")

# ── Machine-learning ─────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "ml", "model.pkl")
SCALER_PATH     = os.path.join(os.path.dirname(__file__), "ml", "scaler.pkl")
LABEL_MAP       = {0: "empty", 1: "occupied", 2: "high_activity"}

# ── LLM settings ─────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL       = "gpt-4o-mini"
LLM_MAX_TOKENS  = 256
LLM_TEMPERATURE = 0.3

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST  = "0.0.0.0"
DASHBOARD_PORT  = 5000
DASHBOARD_DEBUG = False
