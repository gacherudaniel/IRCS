"""
Actuator controller for the IRCS – drives the servo (ventilation damper),
LED (illumination) and buzzer (caregiver alert).

Key behaviours
--------------
* Context-driven targets: each of the four ML context states maps to an
  evidence-based set of environmental targets (temperature, humidity, lux).
* Gradual ramps: when the context state changes, actuator commands move
  toward the new target incrementally over ACTUATOR_RAMP_SECONDS (15 min)
  to avoid abrupt environmental changes that could disturb sleep.
* Safety override layer (always active): hard limits on temperature, CO2,
  and temperature rate-of-change immediately override ML-driven commands and
  activate the buzzer to alert caregivers.

Hardware mapping (BCM):
  GPIO 18 → MG90S servo signal (50 Hz PWM)  – ventilation damper
  GPIO 21 → LED anode via 330Ω              – room illumination
  GPIO 23 → NPN base via 1 kΩ → buzzer      – audible caregiver alert
"""

import logging
import time

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

from config import (
    SERVO_PIN, LED_PIN, BUZZER_PIN,
    SERVO_PWM_FREQ, SERVO_DUTY_STOP, SERVO_DUTY_SLOW, SERVO_DUTY_FULL,
    COMFORT_TARGETS,
    SAFETY_TEMP_MIN_ELDERLY, SAFETY_TEMP_MAX,
    SAFETY_CO2_MAX_PPM, SAFETY_TEMP_DROP_RATE,
    ACTUATOR_RAMP_SECONDS, SENSOR_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)

# How much the servo duty changes per sensor cycle during a ramp
_RAMP_STEP = (SERVO_DUTY_FULL - SERVO_DUTY_STOP) / (
    ACTUATOR_RAMP_SECONDS / max(SENSOR_POLL_INTERVAL, 1)
)


class ActuatorController:
    def __init__(self) -> None:
        self._servo_duty   = SERVO_DUTY_STOP
        self._target_duty  = SERVO_DUTY_STOP
        self._led_on       = False
        self._buzzer_on    = False
        self._last_temp    = None
        self._last_temp_ts = None
        self._safety_active = False

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

    # ── Low-level drivers ─────────────────────────────────────────────────────

    def _write_servo(self, duty: float) -> None:
        duty = max(SERVO_DUTY_STOP, min(SERVO_DUTY_FULL, duty))
        self._servo_duty = duty
        if _GPIO_AVAILABLE and self._servo_pwm:
            self._servo_pwm.ChangeDutyCycle(duty)

    def set_led(self, state: bool) -> None:
        self._led_on = state
        if _GPIO_AVAILABLE:
            GPIO.output(LED_PIN, GPIO.HIGH if state else GPIO.LOW)

    def set_buzzer(self, state: bool) -> None:
        if state == self._buzzer_on:
            return
        self._buzzer_on = state
        if _GPIO_AVAILABLE:
            GPIO.output(BUZZER_PIN, GPIO.HIGH if state else GPIO.LOW)
        if state:
            logger.warning("BUZZER ACTIVATED – safety alert triggered.")

    # ── Safety override ───────────────────────────────────────────────────────

    def _check_safety(self, reading: dict) -> bool:
        """
        Evaluate hard-limit conditions.  Returns True if a safety override
        is active (caller must skip normal policy and maximise ventilation).
        """
        temp    = reading.get("temperature", 22.0)
        co2     = reading.get("co2_ppm",     400)
        now     = time.monotonic()
        alert   = False

        # Temperature floor
        if temp < SAFETY_TEMP_MIN_ELDERLY:
            logger.warning("SAFETY: temp %.1f°C below floor %.1f°C", temp, SAFETY_TEMP_MIN_ELDERLY)
            alert = True

        # Temperature ceiling
        if temp > SAFETY_TEMP_MAX:
            logger.warning("SAFETY: temp %.1f°C above ceiling %.1f°C", temp, SAFETY_TEMP_MAX)
            alert = True

        # CO2 ceiling
        if co2 > SAFETY_CO2_MAX_PPM:
            logger.warning("SAFETY: CO2 %d ppm exceeds %d ppm limit", co2, SAFETY_CO2_MAX_PPM)
            alert = True

        # Rapid temperature drop
        if self._last_temp is not None and self._last_temp_ts is not None:
            dt_min = (now - self._last_temp_ts) / 60
            if dt_min > 0:
                drop_rate = (self._last_temp - temp) / dt_min
                if drop_rate > SAFETY_TEMP_DROP_RATE:
                    logger.warning("SAFETY: temp dropping at %.2f°C/min", drop_rate)
                    alert = True

        self._last_temp    = temp
        self._last_temp_ts = now
        self._safety_active = alert
        return alert

    # ── Context-driven policy ─────────────────────────────────────────────────

    def apply(self, reading: dict, context_state: str) -> None:
        """
        Compute and apply actuator commands.

        Parameters
        ----------
        reading       : dict of sensor values (temperature, co2_ppm, lux, …)
        context_state : str – one of ROOM_EMPTY, ACTIVE_AWAKE, RESTING, SLEEPING
        """
        safety_override = self._check_safety(reading)

        if safety_override:
            # Maximum ventilation + buzzer alert; ignore ML targets
            self._target_duty = SERVO_DUTY_FULL
            self.set_buzzer(True)
        else:
            self.set_buzzer(False)
            targets = COMFORT_TARGETS.get(context_state, COMFORT_TARGETS["ACTIVE_AWAKE"])

            temp    = reading.get("temperature", 22.0)
            co2     = reading.get("co2_ppm",     400)
            lux     = reading.get("lux",         500.0)

            # Servo target based on thermal + air-quality delta
            if context_state == "ROOM_EMPTY":
                self._target_duty = SERVO_DUTY_STOP
            elif temp > targets["temp"] + 1.5 or co2 > targets["co2_max"]:
                self._target_duty = SERVO_DUTY_FULL
            elif temp > targets["temp"] + 0.5:
                self._target_duty = SERVO_DUTY_SLOW
            else:
                self._target_duty = SERVO_DUTY_STOP

            # LED: on when lux is below the target for this context
            led_needed = (context_state != "ROOM_EMPTY") and (lux < targets["lux"] * 0.8)
            self.set_led(led_needed)

        # ── Gradual servo ramp ────────────────────────────────────────────────
        if self._servo_duty < self._target_duty:
            self._write_servo(min(self._servo_duty + _RAMP_STEP, self._target_duty))
        elif self._servo_duty > self._target_duty:
            self._write_servo(max(self._servo_duty - _RAMP_STEP, self._target_duty))

        logger.info(
            "Actuators – state=%s | servo=%.1f%%→%.1f%% | led=%s | buzzer=%s | safety=%s",
            context_state, self._servo_duty, self._target_duty,
            "ON" if self._led_on else "OFF",
            "ON" if self._buzzer_on else "OFF",
            "ACTIVE" if safety_override else "OK",
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        return {
            "servo_duty":    self._servo_duty,
            "target_duty":   self._target_duty,
            "led_on":        self._led_on,
            "buzzer_on":     self._buzzer_on,
            "safety_active": self._safety_active,
        }

    def cleanup(self) -> None:
        self.set_led(False)
        self.set_buzzer(False)
        if _GPIO_AVAILABLE:
            if self._servo_pwm:
                self._servo_pwm.stop()
            GPIO.cleanup([SERVO_PIN, LED_PIN, BUZZER_PIN])


import logging
