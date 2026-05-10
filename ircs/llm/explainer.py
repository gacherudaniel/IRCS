"""
LLM-based natural language explainer for IRCS room state.

Provider priority (first available key wins):
  1. Groq  – fastest, completely free.  console.groq.com  (no credit card)
  2. Anthropic Claude – $5 free credit.  console.anthropic.com
  3. Pollinations.ai – zero signup, OpenAI-compatible public endpoint.
  4. Local templates – always works offline.

The explainer is called:
  • every LLM_CALL_INTERVAL seconds (default 5 min)
  • immediately whenever the predicted context state changes

Quick setup (Groq, recommended)
---------------------------------
  1. Go to https://console.groq.com  → sign up (free, no card)
  2. API Keys → Create API Key → copy it
  3. Add to ircs/.env:  GROQ_API_KEY=gsk_...
"""

import logging
from datetime import datetime

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_COMPAT = True
except ImportError:
    _OPENAI_COMPAT = False

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an intelligent room-monitoring assistant for a care system looking after "
    "elderly people and infants. Given current sensor readings and the detected context "
    "state, produce a brief (2-3 sentence), plain-English status update for caregivers. "
    "Highlight any comfort or safety concerns and suggest one actionable step if needed. "
    "Use a calm, professional tone."
)

# Local fallback templates keyed by context state
_FALLBACK_TEMPLATES: dict[str, str] = {
    "ROOM_EMPTY": (
        "The room is currently unoccupied. Environmental conditions are being maintained "
        "at standby levels. No immediate action is required."
    ),
    "ACTIVE_AWAKE": (
        "An occupant is detected and active in the room. Ventilation and lighting are "
        "adjusted for comfort during waking hours. Continue monitoring for any changes."
    ),
    "RESTING": (
        "The occupant appears to be resting quietly. The system has adjusted conditions "
        "for a calm environment. Ensure the room temperature remains within the comfort range."
    ),
    "SLEEPING": (
        "The occupant is asleep. The system has reduced ventilation and lighting to "
        "support undisturbed sleep. Check that temperature and CO2 levels remain safe."
    ),
}


def _build_prompt(reading: dict, context_state: str, confidence: float) -> str:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temp = reading.get("temperature", "N/A")
    hum  = reading.get("humidity",    "N/A")
    co2  = reading.get("co2_ppm",     "N/A")
    lux  = reading.get("lux",         "N/A")
    dist = reading.get("distance",    "N/A")
    post = reading.get("posture",     "N/A")
    flow = reading.get("flow_score",  "N/A")

    return (
        f"Timestamp: {ts}\n"
        f"Context state: {context_state} (ML confidence: {confidence:.0%})\n"
        f"Temperature:  {temp} °C\n"
        f"Humidity:     {hum} %\n"
        f"CO2 (est.):   {co2} ppm\n"
        f"Illuminance:  {lux} lux\n"
        f"Distance:     {dist} cm\n"
        f"Posture code: {post}  (−1=unknown, 0=upright, 1=reclined, 2=horizontal)\n"
        f"Motion score: {flow}  (0=still, 1=high motion)\n\n"
        "Please provide a caregiver status update."
    )


# ── Pollinations.ai public endpoint (no API key required) ────────────────────
_POLLINATIONS_BASE = "https://text.pollinations.ai/openai"
_POLLINATIONS_MODEL = "openai"   # routes to gpt-4o-mini equivalent


class LLMExplainer:
    """
    Tries LLM providers in priority order:
      Groq (free key) → Anthropic (paid key) → Pollinations (no key) → templates
    """

    def __init__(self) -> None:
        self._mode   = "fallback"
        self._client = None

        if _OPENAI_COMPAT and GROQ_API_KEY:
            self._client = _OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            self._model = GROQ_MODEL
            self._mode  = "groq"
            logger.info("LLM provider: Groq (%s)", GROQ_MODEL)

        elif _ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
            self._client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self._model  = LLM_MODEL
            self._mode   = "anthropic"
            logger.info("LLM provider: Anthropic (%s)", LLM_MODEL)

        elif _OPENAI_COMPAT:
            # Pollinations.ai – zero signup, completely free
            self._client = _OpenAI(
                api_key="pollinations",          # any non-empty string works
                base_url=_POLLINATIONS_BASE,
            )
            self._model = _POLLINATIONS_MODEL
            self._mode  = "pollinations"
            logger.info("LLM provider: Pollinations.ai (no API key required)")

        else:
            logger.warning(
                "No LLM provider available – install 'openai' package or set "
                "GROQ_API_KEY in .env.  Using local fallback templates."
            )

    def explain(self, reading: dict, context_state: str, confidence: float = 1.0) -> str:
        """
        Generate a natural-language caregiver update.

        Parameters
        ----------
        reading       : latest sensor reading dict
        context_state : one of ROOM_EMPTY, ACTIVE_AWAKE, RESTING, SLEEPING
        confidence    : ML classifier confidence (0-1)

        Returns
        -------
        str – plain-English status message
        """
        if self._client is None or self._mode == "fallback":
            return _FALLBACK_TEMPLATES.get(context_state, _FALLBACK_TEMPLATES["ROOM_EMPTY"])

        prompt = _build_prompt(reading, context_state, confidence)
        try:
            if self._mode == "anthropic":
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=LLM_MAX_TOKENS,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()
            else:  # groq or pollinations (both OpenAI-compatible)
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=LLM_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM API call failed (%s): %s – using fallback.", self._mode, exc)
            return _FALLBACK_TEMPLATES.get(context_state, _FALLBACK_TEMPLATES["ROOM_EMPTY"])
