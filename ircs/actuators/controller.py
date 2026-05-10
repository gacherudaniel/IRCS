"""
Actuator controller – drives the servo (ventilation), LED (lighting) and
buzzer (alert) GPIO outputs based on current sensor readings and the
ML-predicted room state.

Hardware mapping (BCM):
  GPIO 18 → MG90S servo signal (50 Hz PWM)   – ventilation / damper
  GPIO 21 → LED anode via 330Ω resistor       – room illumination
  GPIO 23 → NPN base via 1 kΩ → buzzer        – audible alert
"""

import logging

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

from config import (
    SERVO_PIN,
    LED_PIN,
    BUZZER_PIN,
    SERVO_PWM_FREQ,
    SERVO_DUTY_STOP,
    SERVO_DUTY_SLOW,
    SERVO_DUTY_FULL,
    TEMP_HIGH_THRESHOLD,
    TEMP_LOW_THRESHOLD,
    HUMIDITY_HIGH,
    CO2_HIGH_THRESHOLD,
    LDR_DARK_THRESHOLD,
    DISTANCE_OCCUPIED_CM,
)

logger = logging.getLogger(__name__)


class ActuatorController:
    """
    Controls three actuators:
      - Servo  (GPIO 18 / 50 Hz PWM) – ventilation damper (MG90S)
      - LED    (GPIO 21 digital out)  – room illumination
      - Buzzer (GPIO 23 digital out)  – audible alert via NPN transistor
    """

    def __init__(self) -> None:
        self._servo_duty  = SERVO_DUTY_STOP
        self._led_on      = False
        self._buzzer_on   = False

        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in (LED_PIN, BUZZER_PIN):
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(SERVO_PIN, GPIO.OUT)

            self._servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_PWM_FREQ)
            self._servo_pwm.start(SERVO_DUTY_STOP)
        else:
            logger.warning("RPi.GPIO not available – ActuatorController in simulation mode.")
            self._servo_pwm = None

    # ── Individual actuator controls ──────────────────────────────────────────

    def set_fan(self, duty_cycle: float) -> None:
        """Set fan speed (0-100 %). Clamps to valid range."""
        duty_cycle = max(0.0, min(100.0, float(duty_cycle)))
        self._fan_duty = duty_cycle
        if _GPIO_AVAILABLE and self._fan_pwm:
            self._fan_pwm.ChangeDutyCycle(duty_cycle)
        logger.debug("Fan duty cycle → %.0f%%", duty_cycle)

    def set_light(self, state: bool) -> None:
        """Turn room light on (True) or off (False)."""
        self._light_on = state
        if _GPIO_AVAILABLE:
            GPIO.output(LIGHT_PIN, GPIO.HIGH if state else GPIO.LOW)
        logger.debug("Light → %s", "ON" if state else "OFF")

    def set_hvac(self, state: bool) -> None:
        """Activate (True) or deactivate (False) the HVAC relay."""
        self._hvac_on = state
        if _GPIO_AVAILABLE:
            GPIO.output(HVAC_PIN, GPIO.HIGH if state else GPIO.LOW)
        logger.debug("HVAC → %s", "ON" if state else "OFF")

    # ── Policy application ────────────────────────────────────────────────────

    def apply(self, reading: dict, room_state: str) -> None:
        """
        Decide actuator states from sensor readings and classified room state.

        Parameters
        ----------
        reading    : dict with keys temperature, humidity, air_quality, ldr, occupancy
        room_state : str  – one of 'empty', 'occupied', 'high_activity'
        """
        temp        = reading.get("temperature", 22.0)
        humidity    = reading.get("humidity",    50.0)
        air_quality = reading.get("air_quality", 400)
        ldr         = reading.get("ldr",         50000)  # lux value
        occupied    = reading.get("occupancy",   False)

        # ── Servo / ventilation policy ────────────────────────────────────────
        if room_state == "empty":
            servo_duty = SERVO_DUTY_STOP
        elif room_state == "high_activity":
            servo_duty = SERVO_DUTY_FULL
        elif temp > TEMP_HIGH_THRESHOLD or humidity > HUMIDITY_HIGH or air_quality > CO2_HIGH_THRESHOLD:
            servo_duty = SERVO_DUTY_SLOW
        else:
            servo_duty = SERVO_DUTY_STOP

        self.set_servo(servo_duty)

        # ── LED lighting policy ───────────────────────────────────────────────
        led_needed = occupied and (ldr < LDR_DARK_THRESHOLD)
        self.set_led(led_needed)

        # ── Buzzer alert policy ───────────────────────────────────────────────
        # Alert on hazardous air quality or temperature extremes
        alert = occupied and (
            air_quality > CO2_HIGH_THRESHOLD
            or temp > TEMP_HIGH_THRESHOLD
            or temp < TEMP_LOW_THRESHOLD
        )
        self.set_buzzer(alert)

        logger.info(
            "Actuators applied – state=%s | servo=%.1f%% | led=%s | buzzer=%s",
            room_state, servo_duty,
            "ON" if led_needed else "OFF",
            "ON" if alert else "OFF",
        )

    @property
    def status(self) -> dict:
        return {
            "servo_duty":  self._servo_duty,
            "led_on":      self._led_on,
            "buzzer_on":   self._buzzer_on,
        }

    def cleanup(self) -> None:
        if _GPIO_AVAILABLE:
            if self._servo_pwm:
                self._servo_pwm.stop()
            GPIO.cleanup([SERVO_PIN, LED_PIN, BUZZER_PIN])
        if _GPIO_AVAILABLE:
            if self._servo_pwm:
                self._servo_pwm.stop()
            GPIO.cleanup([SERVO_PIN, LED_PIN, BUZZER_PIN])
