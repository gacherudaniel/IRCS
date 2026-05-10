"""
Ultrasonic distance sensor (HC-SR04).
Uses RPi.GPIO with a software timeout to prevent blocking on missed echo.
"""

import time
import logging

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

from config import ULTRASONIC_TRIG_PIN, ULTRASONIC_ECHO_PIN

logger = logging.getLogger(__name__)

_SPEED_OF_SOUND_CM_S = 34300   # cm/s at ~20 °C
_TIMEOUT_S           = 0.04    # 40 ms max round-trip (≈ 6.8 m range)
_TRIG_PULSE_S        = 0.00001 # 10 µs trigger pulse


class UltrasonicSensor:
    def __init__(self) -> None:
        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(ULTRASONIC_TRIG_PIN, GPIO.OUT)
            GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
            GPIO.output(ULTRASONIC_TRIG_PIN, False)
            time.sleep(0.05)   # settle
        else:
            logger.warning("RPi.GPIO not available – UltrasonicSensor in simulation mode.")

    def read_distance(self) -> float:
        """Return distance in centimetres, or -1.0 on timeout/error."""
        if not _GPIO_AVAILABLE:
            import random
            return round(random.uniform(50, 300), 1)

        try:
            # Send 10 µs trigger pulse
            GPIO.output(ULTRASONIC_TRIG_PIN, True)
            time.sleep(_TRIG_PULSE_S)
            GPIO.output(ULTRASONIC_TRIG_PIN, False)

            # Wait for echo HIGH
            start = time.time()
            while GPIO.input(ULTRASONIC_ECHO_PIN) == 0:
                if time.time() - start > _TIMEOUT_S:
                    return -1.0
            pulse_start = time.time()

            # Wait for echo LOW
            while GPIO.input(ULTRASONIC_ECHO_PIN) == 1:
                if time.time() - pulse_start > _TIMEOUT_S:
                    return -1.0
            pulse_end = time.time()

            distance = ((pulse_end - pulse_start) * _SPEED_OF_SOUND_CM_S) / 2
            return round(distance, 1)

        except Exception as exc:
            logger.error("UltrasonicSensor error: %s", exc)
            return -1.0

    def cleanup(self) -> None:
        if _GPIO_AVAILABLE:
            GPIO.cleanup([ULTRASONIC_TRIG_PIN, ULTRASONIC_ECHO_PIN])
