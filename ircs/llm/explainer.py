"""
LLM Explainer – uses the OpenAI Chat Completions API to generate a
concise, human-readable explanation of the current room state and the
automated decisions taken by the IRCS actuator controller.

If the API key is not set or the request fails, a fallback rule-based
explanation is returned so the system remains operational offline.
"""

import logging
from functools import lru_cache

from config import OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert building-automation assistant embedded in an "
    "Intelligent Room Control System (IRCS) running on a Raspberry Pi. "
    "Given real-time sensor data and the ML-predicted room state, write a "
    "single concise paragraph (≤ 60 words) that explains what is happening "
    "in the room and why the automated systems have responded as they have. "
    "Use plain, non-technical language suitable for an end-user dashboard."
)

_FALLBACK_TEMPLATES = {
    "empty":         "The room is currently empty. All non-essential systems have been "
                     "switched off to conserve energy.",
    "occupied":      "The room is occupied. Lighting and climate control are active to "
                     "maintain a comfortable environment.",
    "high_activity": "High activity has been detected. Ventilation is running at full "
                     "speed and all comfort systems are engaged.",
}


def _build_user_prompt(reading: dict, room_state: str) -> str:
    return (
        f"Room state: {room_state}\n"
        f"Temperature: {reading.get('temperature', 'N/A')} °C\n"
        f"Humidity: {reading.get('humidity', 'N/A')} %\n"
        f"Pressure: {reading.get('pressure', 'N/A')} hPa\n"
        f"Air quality (ppm-eq): {reading.get('air_quality', 'N/A')}\n"
        f"Light level (ADC): {reading.get('ldr', 'N/A')}\n"
        f"Distance to nearest object: {reading.get('distance', 'N/A')} cm\n"
        f"Camera occupancy: {'yes' if reading.get('occupancy') else 'no'}\n"
        "Provide the explanation now."
    )


class LLMExplainer:
    def __init__(self) -> None:
        self._client = None
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialised (model=%s).", LLM_MODEL)
            except ImportError:
                logger.warning("openai package not installed – LLM disabled.")
        else:
            logger.warning("OPENAI_API_KEY not set – LLM explainer will use fallback.")

    def explain(self, reading: dict, room_state: str) -> str:
        """
        Generate a natural-language explanation for the current room state.

        Parameters
        ----------
        reading    : dict of raw sensor values
        room_state : str – ML predicted label

        Returns
        -------
        str – explanation text
        """
        if self._client is None:
            return _FALLBACK_TEMPLATES.get(
                room_state,
                f"Room state is '{room_state}'. Automated systems are responding accordingly.",
            )

        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(reading, room_state)},
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            return _FALLBACK_TEMPLATES.get(room_state, "Explanation unavailable.")
