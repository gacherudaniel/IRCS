"""
SQLite logger for IRCS sensor readings and room-state predictions.

Thread-safety: uses a per-connection lock so multiple threads can call
log() concurrently without corruption.
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from config import DB_PATH

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sensor_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    temperature  REAL,
    pressure     REAL,
    altitude     REAL,
    air_quality  INTEGER,
    ldr          REAL,
    flow_score   REAL,
    label        TEXT,
    explanation  TEXT
);
"""

_INSERT_SQL = """
INSERT INTO sensor_log
    (timestamp, temperature, pressure, altitude,
     air_quality, ldr, flow_score, label, explanation)
VALUES
    (:timestamp, :temperature, :pressure, :altitude,
     :air_quality, :ldr, :flow_score, :label, :explanation);
"""


class DatabaseLogger:
    def __init__(self) -> None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()
        logger.info("Database initialised at %s", DB_PATH)

    def log(
        self,
        reading: dict,
        label: str = "",
        explanation: str = "",
    ) -> None:
        """Insert one sensor reading row."""
        row: dict[str, Any] = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "temperature": reading.get("temperature"),
            "pressure":    reading.get("pressure"),
            "altitude":    reading.get("altitude"),
            "air_quality": reading.get("air_quality"),
            "ldr":         reading.get("ldr"),
            "flow_score":  reading.get("flow_score"),
            "label":       label,
            "explanation": explanation,
        }
        with self._lock:
            try:
                self._conn.execute(_INSERT_SQL, row)
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error("Database write error: %s", exc)

    def fetch_recent(self, n: int = 50) -> list[dict]:
        """Return the n most recent log rows as a list of dicts."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sensor_log ORDER BY id DESC LIMIT ?", (n,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_range(self, start: str, end: str) -> list[dict]:
        """Return rows within an ISO-8601 timestamp range."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sensor_log WHERE timestamp BETWEEN ? AND ?"
                " ORDER BY id ASC",
                (start, end),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
