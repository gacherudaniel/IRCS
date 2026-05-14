"""
DHT11 temperature & humidity sensor.
Uses the Adafruit CircuitPython DHT library.

Hardware: DHT11 DATA → GPIO 4 (Pin 7).  3.3 V supply, internal 10 kΩ pull-up
on the DATA line (many DHT11 breakout boards include this).

Note: DHT11 minimum sampling interval is 1 s; accuracy is ±2 °C / ±5 % RH.
"""

import time
import logging

try:
    import board
    import adafruit_dht
    _DHT_AVAILABLE = True
except ImportError:
    _DHT_AVAILABLE = False

from config import DHT11_PIN

logger = logging.getLogger(__name__)

_MAX_RETRIES  = 3
_RETRY_DELAY  = 1.0   # seconds – DHT11 minimum sampling interval is 1 s


class DHT11Sensor:
    def __init__(self) -> None:
        if _DHT_AVAILABLE:
            pin = getattr(board, f"D{DHT11_PIN}", None)
            if pin is None:
                raise ValueError(f"Invalid board pin D{DHT11_PIN}")
            self._device = adafruit_dht.DHT11(pin, use_pulseio=False)
        else:
            logger.warning("adafruit_dht not available – DHT11Sensor in simulation mode.")
            self._device = None

    def _read_raw(self) -> tuple[float, float]:
        """Return (temperature_C, humidity_%) with retry logic."""
        for attempt in range(_MAX_RETRIES):
            try:
                temp = self._device.temperature
                hum  = self._device.humidity
                if temp is not None and hum is not None:
                    return float(temp), float(hum)
            except Exception:
                # DHT sensors occasionally return bad reads; retry
                pass
            time.sleep(_RETRY_DELAY)
        raise RuntimeError("DHT11 failed to return a valid reading after retries.")

    def read_temperature(self) -> float:
        """Return ambient temperature in °C."""
        if not _DHT_AVAILABLE:
            import random
            return round(random.uniform(18.0, 35.0), 1)
        temp, _ = self._read_raw()
        return temp

    def read_humidity(self) -> float:
        """Return relative humidity in %."""
        if not _DHT_AVAILABLE:
            import random
            return round(random.uniform(30.0, 80.0), 1)
        _, hum = self._read_raw()
        return hum

    def read(self) -> dict:
        """Return both readings in a single bus access."""
        if not _DHT_AVAILABLE:
            return {"temperature": self.read_temperature(), "humidity": self.read_humidity()}
        temp, hum = self._read_raw()
        return {"temperature": temp, "humidity": hum}

    def cleanup(self) -> None:
        if self._device is not None:
            self._device.exit()
