"""
IRCS – Sensor Hardware Test Script
Run from the ircs/ directory:  python test_sensors.py

Tests each sensor in isolation and prints a PASS / FAIL summary.
Simulation mode is used automatically when hardware libraries are absent.
"""

import sys
import time
import logging
import traceback

logging.basicConfig(
    level=logging.WARNING,          # suppress sensor library noise during tests
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

PASS  = "\033[32mPASS\033[0m"
FAIL  = "\033[31mFAIL\033[0m"
SKIP  = "\033[33mSKIP\033[0m"
WARN  = "\033[33mWARN\033[0m"
WIDTH = 40

results: list[tuple[str, str, str]] = []   # (sensor, status, detail)


def _row(name: str, status: str, detail: str = "") -> None:
    print(f"  {name:<{WIDTH}} {status}  {detail}")
    results.append((name, status, detail))


def _header(title: str) -> None:
    print(f"\n{'─' * (WIDTH + 20)}")
    print(f"  {title}")
    print(f"{'─' * (WIDTH + 20)}")


# ── DHT11 (temperature & humidity) ──────────────────────────────────────────

def test_dht11() -> None:
    _header("DHT11 – Temperature & Humidity")
    try:
        from sensors.dht22 import DHT11Sensor
        sensor = DHT11Sensor()
        temp = sensor.read_temperature()
        hum  = sensor.read_humidity()
        assert isinstance(temp, float), "Temperature not a float"
        assert isinstance(hum, float),  "Humidity not a float"
        assert -40 <= temp <= 80,  f"Temperature out of range: {temp}"
        assert   0 <= hum  <= 100, f"Humidity out of range: {hum}"
        _row("read_temperature()", PASS, f"{temp:.1f} °C")
        _row("read_humidity()",    PASS, f"{hum:.1f} %RH")
    except Exception as exc:
        _row("DHT11Sensor", FAIL, str(exc))
        traceback.print_exc()


# ── BMP280 (pressure & altitude) ─────────────────────────────────────────────

def test_bmp280() -> None:
    _header("BMP280 – Pressure & Altitude")
    try:
        from sensors.bmp280_sensor import BMP280Sensor
        sensor = BMP280Sensor()
        pressure = sensor.read_pressure()
        altitude = sensor.read_altitude()
        assert isinstance(pressure, float), "Pressure not a float"
        assert isinstance(altitude, float), "Altitude not a float"
        assert 800 <= pressure <= 1100, f"Pressure out of range: {pressure}"
        _row("read_pressure()", PASS, f"{pressure:.2f} hPa")
        _row("read_altitude()", PASS, f"{altitude:.1f} m")
    except Exception as exc:
        _row("BMP280Sensor", FAIL, str(exc))
        traceback.print_exc()


# ── ADS1115 ADC (all 4 channels) ─────────────────────────────────────────────

def test_adc() -> None:
    _header("ADS1115 ADC – Channels A0-A3")
    try:
        from sensors.adc import ADCSensor
        sensor = ADCSensor()
        for ch in range(4):
            raw = sensor.read_channel(ch)
            assert isinstance(raw, int), f"CH{ch}: value not int"
            assert 0 <= raw <= 32767,    f"CH{ch}: raw out of range: {raw}"
            _row(f"read_channel({ch})", PASS, f"raw={raw}")
        voltage = sensor.read_voltage(0)
        assert 0.0 <= voltage <= 4.096, f"Voltage out of range: {voltage}"
        _row("read_voltage(0)",   PASS, f"{voltage:.4f} V")
    except Exception as exc:
        _row("ADCSensor", FAIL, str(exc))
        traceback.print_exc()


# ── MQ-135 Air Quality ───────────────────────────────────────────────────────

def test_air_quality() -> None:
    _header("MQ-135 – Air Quality")
    try:
        from sensors.air_quality import AirQualitySensor
        sensor = AirQualitySensor()
        raw   = sensor.read_raw()
        ppm   = sensor.read_ppm()
        level = sensor.read_level()
        high  = sensor.is_high()
        assert isinstance(raw,   int),  "raw not int"
        assert isinstance(ppm,   int),  "ppm not int"
        assert level in ("good", "moderate", "poor", "hazardous"), \
            f"Unexpected level: {level}"
        assert isinstance(high, bool),  "is_high() not bool"
        _row("read_raw()",   PASS, f"{raw}")
        _row("read_ppm()",   PASS, f"{ppm} ppm")
        _row("read_level()", PASS, f'"{level}"')
        _row("is_high()",    PASS, f"{high}")
    except Exception as exc:
        _row("AirQualitySensor", FAIL, str(exc))
        traceback.print_exc()


# ── LDR (light level) ────────────────────────────────────────────────────────

def test_ldr() -> None:
    _header("LDR – Light Level")
    try:
        from sensors.ldr import LDRSensor
        sensor = LDRSensor()
        raw  = sensor.read_raw()
        lux  = sensor.read_lux()
        dark = sensor.is_dark()
        assert isinstance(raw, int),   "raw not int"
        assert isinstance(lux, float), "lux not float"
        assert lux >= 0.0,             f"lux negative: {lux}"
        assert isinstance(dark, bool), "is_dark() not bool"
        _row("read_raw()",  PASS, f"{raw}")
        _row("read_lux()",  PASS, f"{lux:.2f} lux")
        _row("is_dark()",   PASS, f"{dark}")
    except Exception as exc:
        _row("LDRSensor", FAIL, str(exc))
        traceback.print_exc()


# ── HC-SR04 Ultrasonic ───────────────────────────────────────────────────────

def test_ultrasonic() -> None:
    _header("HC-SR04 – Ultrasonic Distance")
    try:
        from sensors.ultrasonic import UltrasonicSensor
        sensor = UltrasonicSensor()
        dist = sensor.read_distance()
        assert isinstance(dist, float), "distance not float"
        assert dist == -1.0 or 2 <= dist <= 700, \
            f"Distance out of expected range: {dist}"
        status = WARN if dist == -1.0 else PASS
        _row("read_distance()", status,
             f"{dist:.1f} cm" if dist != -1.0 else "timeout / no echo")
        if hasattr(sensor, "cleanup"):
            sensor.cleanup()
    except Exception as exc:
        _row("UltrasonicSensor", FAIL, str(exc))
        traceback.print_exc()


# ── Camera ───────────────────────────────────────────────────────────────────

def test_camera() -> None:
    _header("Camera – Posture & Optical Flow")
    try:
        from sensors.camera import CameraSensor
        sensor = CameraSensor()
        result = sensor.analyse()
        assert "posture"    in result, "Missing 'posture' key"
        assert "flow_score" in result, "Missing 'flow_score' key"
        posture    = result["posture"]
        flow_score = result["flow_score"]
        assert posture in (-1, 0, 1, 2),       f"Invalid posture: {posture}"
        assert 0.0 <= flow_score <= 1.0,        f"flow_score out of range: {flow_score}"
        posture_labels = {-1: "UNKNOWN", 0: "UPRIGHT", 1: "RECLINED", 2: "HORIZONTAL"}
        _row("analyse() → posture",    PASS, posture_labels.get(posture, str(posture)))
        _row("analyse() → flow_score", PASS, f"{flow_score:.3f}")
        if hasattr(sensor, "cleanup"):
            sensor.cleanup()
    except Exception as exc:
        _row("CameraSensor", FAIL, str(exc))
        traceback.print_exc()


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary() -> None:
    print(f"\n{'═' * (WIDTH + 20)}")
    print("  SUMMARY")
    print(f"{'═' * (WIDTH + 20)}")
    passed = sum(1 for _, s, _ in results if s == PASS)
    warned = sum(1 for _, s, _ in results if s == WARN)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    total  = len(results)
    print(f"  Total checks : {total}")
    print(f"  {PASS}         : {passed}")
    print(f"  {WARN}         : {warned}")
    print(f"  {FAIL}         : {failed}")
    print(f"{'═' * (WIDTH + 20)}\n")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    print("\nIRCS Sensor Test Suite")
    print(f"{'═' * (WIDTH + 20)}")

    test_dht11()
    test_bmp280()
    test_adc()
    test_air_quality()
    test_ldr()
    test_ultrasonic()
    test_camera()

    print_summary()
