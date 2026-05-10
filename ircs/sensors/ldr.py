"""
LDR (Light-Dependent Resistor) sensor via ADS1115 (I2C, 16-bit ADC).
Reads raw ADS1115 counts from channel A1 and maps them to a lux estimate.
"""

import logging
import math

from sensors.adc import ADCSensor
from config import LDR_ADC_CHANNEL, LDR_DARK_THRESHOLD

logger = logging.getLogger(__name__)

# ADS1115 full scale (gain=1)
_ADS1115_MAX = 32767


class LDRSensor:
    """
    The LDR forms a voltage divider with a fixed resistor connected to the
    ADS1115 A1 input.  Higher ADC value → more light.
    """

    def __init__(self) -> None:
        self._adc = ADCSensor()

    def read_raw(self) -> int:
        """Return raw ADC counts (0-1023)."""
        return self._adc.read_channel(LDR_ADC_CHANNEL)

    def read_lux(self) -> float:
        """
        Return an approximate lux value using a logarithmic curve fit.
        ADS1115 raw 0-32767 → lux range ~1-100 000 lux.
        The mapping is approximate and depends on the LDR model and
        voltage-divider resistor value.
        """
        raw = self.read_raw()
        if raw == 0:
            return 0.0
        # Approximate: lux ≈ 10^((raw/32767)*5)  → range 1–100 000 lux
        lux = 10 ** ((raw / _ADS1115_MAX) * 5)
        return round(lux, 2)

    def is_dark(self) -> bool:
        """Return True when lux level is below the configured threshold."""
        return self.read_lux() < LDR_DARK_THRESHOLD

    def cleanup(self) -> None:
        self._adc.cleanup()
