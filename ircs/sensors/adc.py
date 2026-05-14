"""
ADS1115 ADC driver (I2C, 16-bit, 4 single-ended channels).
Replaces the original MCP3008 SPI driver to match the hardware wiring
where ADS1115 shares the I2C bus with BMP280 and the OLED display.

Channel assignments (see config.py):
  A0 → MQ-135 air-quality sensor
  A1 → LDR
  A2/A3 → spare
"""

import logging

try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    _ADS_AVAILABLE = True
    _CHANNELS = (0, 1, 2, 3)
except ImportError:
    _ADS_AVAILABLE = False
    _CHANNELS = (0, 1, 2, 3)

from config import ADS1115_I2C_ADDRESS, ADS1115_GAIN

logger = logging.getLogger(__name__)

# ADS1115 full-scale voltage for gain=1 (±4.096 V) → maps to raw ±32767
_ADS1115_MAX_RAW  = 32767
_ADS1115_VREF     = 4.096   # V (full-scale for gain=1)


class ADCSensor:
    """16-bit I2C ADC (ADS1115). Reads single-ended channels A0-A3."""

    def __init__(self) -> None:
        if _ADS_AVAILABLE:
            i2c = busio.I2C(board.SCL, board.SDA)
            self._ads = ADS.ADS1115(i2c, address=ADS1115_I2C_ADDRESS,
                                    gain=ADS1115_GAIN)
            # Pre-create AnalogIn objects for each channel
            self._channels = [AnalogIn(self._ads, ch) for ch in _CHANNELS]
        else:
            logger.warning("adafruit_ads1x15 not available – ADCSensor in simulation mode.")
            self._ads      = None
            self._channels = []

    def read_channel(self, channel: int) -> int:
        """
        Return raw ADS1115 value (0-32767) for the given channel (0-3).
        Negative voltages are clamped to 0 (sensor outputs are unipolar).
        """
        if not (0 <= channel <= 3):
            raise ValueError(f"Channel must be 0-3, got {channel}")

        if not _ADS_AVAILABLE:
            import random
            return random.randint(0, 32767)

        raw = self._channels[channel].value
        return max(0, raw)

    def read_voltage(self, channel: int) -> float:
        """Return the measured voltage in volts (0.0-3.3 V)."""
        if not _ADS_AVAILABLE:
            import random
            return round(random.uniform(0.0, 3.3), 4)
        return round(self._channels[channel].voltage, 4)

    def read_normalised(self, channel: int) -> float:
        """Return a 0.0-1.0 fraction of the full-scale range."""
        raw = self.read_channel(channel)
        return raw / _ADS1115_MAX_RAW

    def cleanup(self) -> None:
        pass   # I2C bus is shared; do not close it here

