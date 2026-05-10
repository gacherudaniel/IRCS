"""
BMP280 barometric pressure & temperature sensor (I2C).
Uses the Adafruit CircuitPython BMP280 library.
"""

import logging

try:
    import board
    import busio
    import adafruit_bmp280
    _BMP_AVAILABLE = True
except ImportError:
    _BMP_AVAILABLE = False

from config import BMP280_I2C_ADDRESS

logger = logging.getLogger(__name__)

_SEA_LEVEL_PRESSURE_HPA = 1013.25   # standard atmosphere


class BMP280Sensor:
    def __init__(self) -> None:
        if _BMP_AVAILABLE:
            i2c = busio.I2C(board.SCL, board.SDA)
            self._device = adafruit_bmp280.Adafruit_BMP280_I2C(
                i2c, address=BMP280_I2C_ADDRESS
            )
            self._device.sea_level_pressure = _SEA_LEVEL_PRESSURE_HPA
        else:
            logger.warning("adafruit_bmp280 not available – BMP280Sensor in simulation mode.")
            self._device = None

    def read_temperature(self) -> float:
        """Return temperature in °C."""
        if not _BMP_AVAILABLE:
            import random
            return round(random.uniform(18.0, 35.0), 2)
        return round(self._device.temperature, 2)

    def read_pressure(self) -> float:
        """Return atmospheric pressure in hPa."""
        if not _BMP_AVAILABLE:
            import random
            return round(random.uniform(995.0, 1025.0), 2)
        return round(self._device.pressure, 2)

    def read_altitude(self) -> float:
        """Return altitude estimate in metres above sea level."""
        if not _BMP_AVAILABLE:
            import random
            return round(random.uniform(0.0, 500.0), 1)
        return round(self._device.altitude, 1)

    def read(self) -> dict:
        return {
            "temperature": self.read_temperature(),
            "pressure":    self.read_pressure(),
            "altitude":    self.read_altitude(),
        }
