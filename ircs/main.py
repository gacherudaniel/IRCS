"""
IRCS – Intelligent Room Control System
Entry point: initialises all subsystems and runs the main control loop.
"""

import time
import logging
import threading

from config import (
    SENSOR_POLL_INTERVAL,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DASHBOARD_DEBUG,
)
from sensors.ultrasonic   import UltrasonicSensor
from sensors.dht22        import DHT11Sensor
from sensors.bmp280_sensor import BMP280Sensor
from sensors.adc          import ADCSensor
from sensors.air_quality  import AirQualitySensor
from sensors.ldr          import LDRSensor
from sensors.camera       import CameraSensor
from ml.feature_extractor import FeatureExtractor
from ml.classifier        import RoomClassifier
from actuators.controller import ActuatorController
from database.logger      import DatabaseLogger
from llm.explainer        import LLMExplainer
from dashboard.app        import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def sensor_loop(
    sensors: dict,
    classifier: RoomClassifier,
    feature_extractor: FeatureExtractor,
    actuator: ActuatorController,
    db_logger: DatabaseLogger,
    llm: LLMExplainer,
    state: dict,
) -> None:
    """Continuously reads sensors, classifies room state and drives actuators."""
    while state.get("running", True):
        try:
            camera_data = sensors["camera"].analyse()
            reading = {
                "temperature": sensors["dht22"].read_temperature(),
                "humidity":    sensors["dht22"].read_humidity(),
                "pressure":    sensors["bmp280"].read_pressure(),
                "altitude":    sensors["bmp280"].read_altitude(),
                "air_quality": sensors["air_quality"].read_ppm(),
                "ldr":         sensors["ldr"].read_lux(),
                "distance":    sensors["ultrasonic"].read_distance(),
                "posture":     camera_data["posture"],
                "flow_score":  camera_data["flow_score"],
            }

            features   = feature_extractor.extract(reading)
            label_id   = classifier.predict(features)
            label_name = classifier.label_name(label_id)

            actuator.apply(reading, label_name)
            db_logger.log(reading, label_name)

            explanation = llm.explain(reading, label_name)

            state["latest"] = {
                "reading":     reading,
                "label":       label_name,
                "explanation": explanation,
            }

            logger.info("State=%s | temp=%.1f°C | hum=%.1f%% | AQ=%d",
                        label_name, reading["temperature"],
                        reading["humidity"], reading["air_quality"])

        except Exception as exc:
            logger.error("Sensor loop error: %s", exc)

        time.sleep(SENSOR_POLL_INTERVAL)


def main() -> None:
    logger.info("Starting IRCS …")

    # ── Initialise sensors ────────────────────────────────────────────────────
    sensors = {
        "ultrasonic":  UltrasonicSensor(),
        "dht22":       DHT11Sensor(),
        "bmp280":      BMP280Sensor(),
        "adc":         ADCSensor(),
        "air_quality": AirQualitySensor(),
        "ldr":         LDRSensor(),
        "camera":      CameraSensor(),
    }

    # ── Initialise ML pipeline ────────────────────────────────────────────────
    feature_extractor = FeatureExtractor()
    classifier        = RoomClassifier()

    # ── Initialise actuator controller ────────────────────────────────────────
    actuator = ActuatorController()

    # ── Initialise database ───────────────────────────────────────────────────
    db_logger = DatabaseLogger()

    # ── Initialise LLM explainer ──────────────────────────────────────────────
    llm = LLMExplainer()

    # Shared state dict (thread-safe for simple reads/writes)
    state = {"running": True, "latest": {}}

    # ── Start sensor loop in background thread ────────────────────────────────
    sensor_thread = threading.Thread(
        target=sensor_loop,
        args=(sensors, classifier, feature_extractor, actuator, db_logger, llm, state),
        daemon=True,
        name="sensor-loop",
    )
    sensor_thread.start()
    logger.info("Sensor loop thread started.")

    # ── Start dashboard (blocks until KeyboardInterrupt) ──────────────────────
    app = create_app(state, db_logger)
    try:
        app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG,
                use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutdown requested – stopping IRCS.")
    finally:
        state["running"] = False
        for sensor in sensors.values():
            if hasattr(sensor, "cleanup"):
                sensor.cleanup()
        actuator.cleanup()
        logger.info("IRCS stopped.")


if __name__ == "__main__":
    main()
