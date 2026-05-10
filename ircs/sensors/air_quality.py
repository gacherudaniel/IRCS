"""
MQ-135 air-quality sensor via ADS1115 (I2C, 16-bit ADC).
Reads raw ADS1115 counts from channel A0 and maps them to a ppm estimate.
"""

import logging

from sensors.adc import ADCSensor
from config import MQ135_ADC_CHANNEL, CO2_HIGH_THRESHOLD

logger = logging.getLogger(__name__)

# Approximate thresholds for MQ-135 in typical indoor environment (16-bit, 0-32767)
_LEVEL_GOOD     = 9800
_LEVEL_MODERATE = 19600
_LEVEL_POOR     = 29400


class AirQualitySensor:
    def __init__(self) -> None:
        self._adc = ADCSensor()

    def read_raw(self) -> int:
        """Return raw ADC counts (0-1023)."""
        return self._adc.read_channel(MQ135_ADC_CHANNEL)

    def read_ppm(self) -> int:
        """Return a scaled ppm-equivalent value (linear approximation)."""
        raw = self.read_raw()
        # Linear scaling: 0-32767 raw → 0-2000 ppm equivalent
        return int((raw / 32767.0) * 2000)

    def read_level(self) -> str:
        """Return a qualitative air-quality label."""
        raw = self.read_raw()
        if raw < _LEVEL_GOOD:
            return "good"
        if raw < _LEVEL_MODERATE:
            return "moderate"
        if raw < _LEVEL_POOR:
            return "poor"
        return "hazardous"

    def is_high(self) -> bool:
        """Return True when air-quality reading exceeds the configured threshold."""
        return self.read_ppm() > CO2_HIGH_THRESHOLD

    def cleanup(self) -> None:
        self._adc.cleanup()
