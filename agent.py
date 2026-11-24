"""Turns model results into plain-language explanations and recommendations.

If ANTHROPIC_API_KEY is set, Claude writes the text from a JSON summary of the facts.
Otherwise, or if the API call fails, simple rules write it. Claude only sees the facts
built by build_context(), and the reason for any fallback is returned to the app.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
UNIT = "Mega Units (MU) per day"
DEFAULT_MODEL = "claude-sonnet-5-5"
SYSTEM_PROMPT = (
    "You are an energy-grid analyst. Using ONLY the JSON facts in the user message, reply with a "
    "single JSON object with two keys: 'explanation' (3 to 5 short strings on why demand varies) and "
    "'recommendations' (3 to 5 actionable strings). Do not write any text outside the JSON. "
    "Do not state any number that is not in the facts."
)


def build_context(
    entity: str,
    series: pd.Series,
    fc_series: pd.Series,
    best: str,
    metrics: dict,
    anomalies: pd.DataFrame,
    corr_temp: float | None = None,
) -> dict:
    """Plain, JSON-serialisable summary of the facts behind the insights."""
    weekday = series.groupby(series.index.day_of_week).mean()
    flagged = anomalies[anomalies["anomaly"]]
    return {
        "entity": entity,
        "unit": UNIT,
        "history_start": str(series.index.min().date()),
        "history_end": str(series.index.max().date()),
        "history_mean": round(float(series.mean()), 1),
        "history_min": {"date": str(series.idxmin().date()), "value": round(float(series.min()), 1)},
        "history_max": {"date": str(series.idxmax().date()), "value": round(float(series.max()), 1)},
        "weekday_avg": {DAYS[int(k)]: round(float(v), 1) for k, v in weekday.items()},
        "best_model": best,
        "holdout_metrics": {k: round(float(v), 2) for k, v in metrics.items()},
        "forecast_next_7_days": [
            {"date": str(d.date()), "value": round(float(v), 1)} for d, v in fc_series.head(7).items()
        ],
        "n_anomalies": int(flagged.shape[0]),
        "recent_anomalies": [
            {"date": str(d.date()), "value": round(float(r["value"]), 1), "z": round(float(r["z"]), 2)}
            for d, r in flagged.tail(5).iterrows()
        ],
        "n_peak_days": int(anomalies["peak"].sum()),
        "corr_with_temp": None if corr_temp is None else round(float(corr_temp), 2),
    }


def rule_based(ctx: dict) -> dict:
    wk = ctx["weekday_avg"]
    hi = max(wk, key=wk.get)
    lo = min(wk, key=wk.get)
    spread_pct = 100 * (wk[hi] - wk[lo]) / wk[lo] if wk[lo] else 0.0
    if spread_pct < 5:
        weekly_text = (f"Weekday averages differ only slightly: highest on {hi} ({wk[hi]}) and lowest on "
                       f"{lo} ({wk[lo]}), a spread of about {spread_pct:.1f}%. A weekly effect is weak here.")
    else:
        weekly_text = (f"Demand varies by weekday: highest on {hi} ({wk[hi]}) and lowest on {lo} ({wk[lo]}), "
                       f"a spread of about {spread_pct:.1f}%.")
    explanation = [
        f"Average demand over the period is {ctx['history_mean']} {ctx['unit']}. "
        f"The highest day was {ctx['history_max']['value']} on {ctx['history_max']['date']}.",
        weekly_text,
    ]
    if ctx.get("corr_with_temp") is not None:
        explanation.append(
            f"Correlation between daily demand and mean temperature is {ctx['corr_with_temp']}. "
            "This includes seasonal effects, so it shows association, not cause."
        )
    explanation.append(
        f"{ctx['n_anomalies']} days were flagged as anomalous. They deviate sharply from their local "
        "weekly level and may reflect outages, holidays or reporting errors."
    )
    if spread_pct < 5:
        recommendations = [
            "Plan capacity around the forecast peak days rather than by weekday, since weekday differences are small.",
            "Check whether maintenance windows can be moved to low-demand days, using the forecast to choose them.",
        ]
    else:
        recommendations = [
            f"Plan extra generation or procurement for {hi}, the weekly high, and for forecast peak days.",
            f"Consider scheduling flexible maintenance or load on {lo}, the weekly low.",
        ]
    recommendations += [
        "Check flagged anomaly dates before using them for planning, since they may be data errors.",
        f"Re-run the forecast regularly. The best model by RMSE on the holdout window is {ctx['best_model']}.",
    ]
    return {"explanation": explanation, "recommendations": recommendations}


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the model reply")
    return json.loads(match.group(0))


def _validate(out: dict) -> dict:
    cleaned = {}
    for key in ("explanation", "recommendations"):
        items = out.get(key)
        if not isinstance(items, list) or not items or not all(isinstance(x, str) and x.strip() for x in items):
            raise ValueError(f"'{key}' must be a non-empty list of strings")
        cleaned[key] = [x.strip() for x in items[:5]]
    return cleaned


def advise(ctx: dict) -> dict:
    """Return {'explanation', 'recommendations', 'source', 'note'}. Never raises."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return {**rule_based(ctx), "source": "rule-based",
                "note": "Set ANTHROPIC_API_KEY to get Claude-written insights."}
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=900,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(ctx)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return {**_validate(_parse_json(text)), "source": "Claude", "note": ""}
    except Exception as exc:
        return {**rule_based(ctx), "source": "rule-based",
                "note": f"Claude request failed ({type(exc).__name__}); showing rule-based text."}
