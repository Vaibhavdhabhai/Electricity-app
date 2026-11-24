"""Chat assistant for the dashboard. Answers questions about the region currently selected.

Provider order, first one with a key set wins:
  1. Google Gemini (free tier)                -> GEMINI_API_KEY (or GOOGLE_API_KEY)
  2. Groq (free tier, no card needed)         -> GROQ_API_KEY
  3. Anthropic Claude                         -> ANTHROPIC_API_KEY
  4. Rule-based answers (no key, no internet) -> always available

Every path receives the facts dictionary built by agent.build_context(). Data questions use only
those facts; general concept questions (RMSE, MAE, SARIMA ...) may use the model's own knowledge.
"""
from __future__ import annotations

import json
import os
import re

import agent

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # override with GEMINI_MODEL
# Tried in order if the main model is retired (404), rate limited (429) or overloaded (503).
GEMINI_FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3.5-flash-lite"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # check Groq's model list; override with GROQ_MODEL
MAX_HISTORY_MESSAGES = 6
# Terms used on the dashboard. Used in the AI prompt and by the offline fallback.
# "rule": False marks generic words that are only explained by the AI, so the offline
# rules do not hijack questions like "what is the peak demand?".
GLOSSARY = [
    {"name": "MU", "terms": ["mu", "million units"], "full": "Million Units",
     "meaning": "the unit of daily electricity energy in this data. 1 MU = 1 million kWh = 1 GWh."},
    {"name": "POSOCO", "terms": ["posoco", "grid-india", "grid india"],
     "full": "Power System Operation Corporation Limited",
     "meaning": "the Indian grid operator (now called Grid-India) that publishes the daily regional and state consumption data used here."},
    {"name": "RMSE", "terms": ["rmse", "root mean squared error", "root mean square error"],
     "full": "Root Mean Squared Error",
     "meaning": "the typical size of forecast errors in MU, with large misses penalised more. Lower is better. It is used to pick the best model."},
    {"name": "MAE", "terms": ["mae", "mean absolute error"], "full": "Mean Absolute Error",
     "meaning": "the average size of forecast errors in MU, treating every miss equally. Lower is better."},
    {"name": "MAPE", "terms": ["mape", "mean absolute percentage error"],
     "full": "Mean Absolute Percentage Error",
     "meaning": "the average forecast error as a percentage of actual demand, so it is easy to compare across regions. Lower is better, but it inflates when actual demand is near zero."},
    {"name": "SARIMA", "terms": ["sarima", "arima"],
     "full": "Seasonal AutoRegressive Integrated Moving Average",
     "meaning": "a statistical model that forecasts from past values, past errors and repeating seasonal patterns such as the weekly cycle."},
    {"name": "Holt-Winters", "terms": ["holt-winters", "holt winters", "holtwinters"],
     "full": "Triple exponential smoothing",
     "meaning": "a model that tracks the level, trend and seasonality of demand, giving more weight to recent days."},
    {"name": "XGBoost / GBM", "terms": ["xgboost", "xgb", "gbm", "gradient boosting"],
     "full": "Extreme Gradient Boosting / Gradient Boosting Machine",
     "meaning": "a machine-learning model that combines many small decision trees. Here it predicts demand from lag features (recent past values)."},
    {"name": "Seasonal Naive", "terms": ["seasonal naive", "seasonal-naive"], "full": "Seasonal Naive",
     "meaning": "the simplest baseline: it forecasts each day as the value from the same weekday one week earlier. Other models should beat it."},
    {"name": "Lag features", "terms": ["lag features", "lag feature", "lags", "lag"], "full": "Lag features",
     "meaning": "past values of demand (for example 1, 7 or 14 days ago) used as inputs so a machine-learning model can learn patterns.", "rule": False},
    {"name": "Z-score", "terms": ["z-score", "z score", "zscore", "z-score threshold"], "full": "Standard score",
     "meaning": "how many standard deviations a day's demand is from normal. A day is flagged as an anomaly when its z-score passes the sidebar threshold (default 3.5)."},
    {"name": "Isolation Forest", "terms": ["isolation forest", "iso forest", "iso_flag"], "full": "Isolation Forest",
     "meaning": "a machine-learning method that flags days that are easy to separate from the rest. It is a second opinion beside the z-score (the iso_flag column)."},
    {"name": "Anomaly", "terms": ["anomaly", "anomalies"], "full": "Anomaly",
     "meaning": "a day whose demand is unusually high or low compared with the pattern, flagged by the z-score and/or Isolation Forest.", "rule": False},
    {"name": "Peak day", "terms": ["peak day", "peak days"], "full": "Peak day",
     "meaning": "a day in the top 5% of demand for the selected region.", "rule": False},
    {"name": "Holdout", "terms": ["holdout", "hold-out", "test window"], "full": "Holdout window",
     "meaning": "the last N days (the forecast horizon) kept aside as a test. Models train on earlier data and are scored on these days.", "rule": False},
    {"name": "Forecast horizon", "terms": ["forecast horizon", "horizon"], "full": "Forecast horizon",
     "meaning": "how many days ahead the models predict, set by the sidebar slider.", "rule": False},
    {"name": "Interpolated days", "terms": ["interpolated", "interpolation"], "full": "Interpolated days",
     "meaning": "days with missing readings that were filled by estimating from neighbouring days. They are not real observations.", "rule": False},
    {"name": "Correlation", "terms": ["correlation", "corr"], "full": "Correlation coefficient",
     "meaning": "a number from -1 to 1 showing how closely two series move together. It shows association, not cause.", "rule": False},
    {"name": "Open-Meteo", "terms": ["open-meteo", "open meteo", "openmeteo"], "full": "Open-Meteo",
     "meaning": "a free weather service the app uses for temperature data when the temperature comparison is switched on."},
    {"name": "NR", "terms": ["nr"], "full": "Northern Region", "meaning": "India's northern regional power grid."},
    {"name": "WR", "terms": ["wr"], "full": "Western Region", "meaning": "India's western regional power grid."},
    {"name": "SR", "terms": ["sr"], "full": "Southern Region", "meaning": "India's southern regional power grid."},
    {"name": "ER", "terms": ["er"], "full": "Eastern Region", "meaning": "India's eastern regional power grid."},
    {"name": "NER", "terms": ["ner"], "full": "North-Eastern Region", "meaning": "India's north-eastern regional power grid."},
]

_EXPLAIN_CUE = re.compile(
    r"\b(what|whats|what's|full form|stands? for|stand for|meaning|mean|means|explain|define|"
    r"definition|abbreviation|expand|expansion|full name|tell me about)\b"
)


def _term_in(text: str, term: str) -> int:
    m = re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text)
    return m.start() if m else -1


def glossary_text() -> str:
    return "\n".join(f"- {g['name']} ({g['full']}): {g['meaning']}" for g in GLOSSARY)


def glossary_answer(question: str, ctx: dict) -> str | None:
    """Offline definitions for dashboard jargon. Returns None when no term is being asked about."""
    q = question.lower()
    if not _EXPLAIN_CUE.search(q):
        return None
    hits = []
    for g in GLOSSARY:
        if g.get("rule", True) is False:
            continue
        pos = [p for p in (_term_in(q, t) for t in g["terms"]) if p >= 0]
        if pos:
            hits.append((min(pos), g))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    lines = [f"**{g['name']}** ({g['full']}): {g['meaning']}" for _, g in hits]
    names = {g["name"] for _, g in hits}
    try:
        m = ctx.get("holdout_metrics") or {}
        shown = [f"{k} {m[k]}" for k in ("RMSE", "MAE", "MAPE") if k in names and k in m]
        if shown:
            lines.append(f"For the best model here ({ctx.get('best_model')}): " + ", ".join(shown) + ".")
    except Exception:
        pass
    return "\n\n".join(lines)


SYSTEM_PROMPT = (
    "You are an assistant inside an electricity demand forecasting dashboard. "
    "For questions about this region's data, forecasts, models or anomalies, use ONLY the FACTS "
    "at the end and do not invent numbers. If the facts do not contain the answer, say so and name "
    "the dashboard tab that might show it (Overview, Forecast, Peaks & Anomalies, AI Insights). "
    "You can also explain anything that appears in the facts or on the dashboard: the full form of "
    "any abbreviation, what a metric, model, unit, field name or region means, and how it is used. "
    "Use the GLOSSARY below and your general knowledge for these, and where the facts contain a "
    "relevant figure, connect the explanation to it. Keep answers to 2 to 4 sentences, or a short "
    "list when several terms are asked about. Interpolated days are not real observations; "
    "mention this when it matters.\n\nGLOSSARY:\n" + glossary_text() + "\n\nFACTS:\n"
)
HELP_TEXT = (
    "I can answer questions about this region's peak and lowest demand, average demand, weekday "
    "pattern, anomalies, the forecast for the next 7 days, the best model, and temperature. "
    "You can also ask what any term means, such as RMSE, MU or SARIMA. "
    "Try: \"Which day had the highest demand?\" or \"Which model is best?\""
)


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


GROQ_HINTS = {
    400: "the request was rejected; check GROQ_MODEL",
    401: "the key is invalid or was copied incorrectly",
    403: "access is blocked for this key or network",
    404: "the model name in GROQ_MODEL was not found; check Groq's model list",
    429: "the free-tier rate limit was reached; wait a minute and try again",
}


GEMINI_HINTS = {
    401: "the key was rejected; check GEMINI_API_KEY",
    400: "the request or key was rejected; check GEMINI_API_KEY and GEMINI_MODEL",
    403: "the key is not allowed for this API; enable the Gemini API for the key in Google AI Studio",
    404: "the model name in GEMINI_MODEL was not found or is no longer available",
    429: "the free-tier rate limit was reached; wait a minute and try again",
    503: "Google's servers are busy; try again in a minute",
}


def describe_error(exc: Exception, hints: dict = GROQ_HINTS) -> str:
    """Short error text for the note, including the HTTP status and a likely cause when known."""
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code is None:
        return type(exc).__name__
    hint = hints.get(code, "")
    return f"{type(exc).__name__} {code}" + (f": {hint}" if hint else "")


def rule_answer(question: str, ctx: dict) -> str:
    """Answers a few common questions without any API call."""
    q = question.lower()
    glossary = glossary_answer(question, ctx)
    if glossary:
        return glossary
    wk = ctx["weekday_avg"]
    if _has(q, "temperature", "weather", "humidity"):
        if ctx.get("corr_with_temp") is None:
            return ("Temperature is not part of this view. Tick \"Compare with temperature\" in the "
                    "sidebar to see its correlation with demand.")
        return (f"Demand and mean temperature have a correlation of {ctx['corr_with_temp']} over the period. "
                "This includes seasonal effects, so it shows association, not cause.")
    if _has(q, "anomal", "unusual", "flag", "abnormal"):
        n = ctx["n_anomalies"]
        if n == 0:
            return "No days were flagged as anomalous at the current threshold."
        recent = ", ".join(f"{a['date']} ({a['value']} MU)" for a in ctx["recent_anomalies"])
        return f"{n} days were flagged as anomalous. The most recent flagged days: {recent}."
    if _has(q, "highest", "peak", "maximum", "max", "biggest"):
        h = ctx["history_max"]
        return (f"The highest demand in the period was {h['value']} MU on {h['date']}. "
                f"There are {ctx['n_peak_days']} peak days (top 5% of demand).")
    if _has(q, "lowest", "minimum", "smallest"):
        lo = ctx["history_min"]
        return f"The lowest demand in the period was {lo['value']} MU on {lo['date']}."
    if _has(q, "weekday", "weekly", "week", "day of the week"):
        hi = max(wk, key=wk.get)
        lo_day = min(wk, key=wk.get)
        return (f"Average demand is highest on {hi} ({wk[hi]} MU) and lowest on {lo_day} ({wk[lo_day]} MU). "
                "The difference is small, so treat the weekly pattern as indicative.")
    if _has(q, "best", "model", "accurate", "accuracy", "rmse", "error", "mae", "mape"):
        m = ctx["holdout_metrics"]
        return (f"The best model by RMSE on the holdout window is {ctx['best_model']}: "
                f"RMSE {m['RMSE']}, MAE {m['MAE']}, MAPE {m['MAPE']}%. The holdout is only the last "
                "few weeks, so small differences between models are not conclusive.")
    if _has(q, "forecast", "next", "tomorrow", "future", "predict"):
        rows = ", ".join(f"{f['date']}: {f['value']} MU" for f in ctx["forecast_next_7_days"])
        return f"Forecast for the next 7 days ({ctx['best_model']}): {rows}."
    if re.search(r"\b(average|mean|typical)\b", q):
        return f"Average demand over the period is {ctx['history_mean']} {ctx['unit']}."
    return HELP_TEXT


def _gemini_reply(system: str, messages: list) -> str:
    import requests

    primary = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    key = os.getenv("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    # Gemini calls the assistant role "model" and takes the system text separately.
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    models_to_try = [primary] + [m for m in GEMINI_FALLBACK_MODELS if m != primary]
    last_exc = None
    for model in models_to_try:
        try:
            resp = requests.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": key.strip(), "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": contents,
                    # Thinking models count reasoning tokens against this limit, so keep headroom.
                    "generationConfig": {"maxOutputTokens": 1500},
                },
                timeout=20,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_exc = exc
            code = exc.response.status_code if exc.response is not None else None
            if code in (404, 429, 503):
                continue  # try the next model
            raise
        candidates = resp.json().get("candidates") or []
        if not candidates:
            raise ValueError("no candidates in reply")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(p.get("text", "")) for p in parts).strip()
        if not text:
            raise ValueError("empty reply")
        return text
    raise last_exc


def _groq_reply(system: str, messages: list) -> str:
    import requests

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            "max_tokens": 500,
            "messages": [{"role": "system", "content": system}] + messages,
        },
        timeout=30,
    )
    resp.raise_for_status()
    text = str(resp.json()["choices"][0]["message"]["content"]).strip()
    if not text:
        raise ValueError("empty reply")
    return text


def _claude_reply(system: str, messages: list) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", agent.DEFAULT_MODEL),
        max_tokens=500,
        system=system,
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    if not text:
        raise ValueError("empty reply")
    return text


def answer(question: str, ctx: dict, history: list | None = None) -> dict:
    """Return {'answer', 'source', 'note'}. Never raises."""
    question = (question or "").strip()
    if not question:
        return {"answer": "Please type a question.", "source": "none", "note": ""}

    system = SYSTEM_PROMPT + json.dumps(ctx)
    past = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-MAX_HISTORY_MESSAGES:]]
    messages = past + [{"role": "user", "content": question}]

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        try:
            return {"answer": _gemini_reply(system, messages), "source": "Gemini", "note": ""}
        except Exception as exc:
            return {"answer": rule_answer(question, ctx), "source": "rule-based",
                    "note": f"Gemini request failed ({describe_error(exc, GEMINI_HINTS)}); showing a rule-based answer."}

    if os.getenv("GROQ_API_KEY"):
        try:
            return {"answer": _groq_reply(system, messages), "source": "Groq", "note": ""}
        except Exception as exc:
            return {"answer": rule_answer(question, ctx), "source": "rule-based",
                    "note": f"Groq request failed ({describe_error(exc)}); showing a rule-based answer."}

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return {"answer": _claude_reply(system, messages), "source": "Claude", "note": ""}
        except Exception as exc:
            return {"answer": rule_answer(question, ctx), "source": "rule-based",
                    "note": f"Claude request failed ({type(exc).__name__}); showing a rule-based answer."}

    return {"answer": rule_answer(question, ctx), "source": "rule-based",
            "note": "Add a free GEMINI_API_KEY or GROQ_API_KEY for free-form answers."}
