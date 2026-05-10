"""
Flask dashboard for the IRCS.

Routes
------
GET  /              – Main dashboard page (HTML)
GET  /api/status    – Latest sensor reading + room state (JSON)
GET  /api/history   – Last N log rows from the database (JSON)
"""

import logging
from flask import Flask, jsonify, render_template, request

from database.logger import DatabaseLogger

logger = logging.getLogger(__name__)


def create_app(state: dict, db_logger: DatabaseLogger) -> Flask:
    """
    Application factory.

    Parameters
    ----------
    state     : shared dict updated by the sensor loop thread
    db_logger : DatabaseLogger instance for history queries
    """
    app = Flask(__name__, template_folder="templates")

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        latest = state.get("latest", {})
        return jsonify(latest)

    @app.route("/api/history")
    def api_history():
        try:
            n = int(request.args.get("n", 50))
            n = max(1, min(n, 500))   # clamp to safe range
        except ValueError:
            n = 50
        rows = db_logger.fetch_recent(n)
        return jsonify(rows)

    # ── Error handlers ────────────────────────────────────────────────────────

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app
