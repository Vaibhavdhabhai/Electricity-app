"""Forecasting models and accuracy metrics.

Every model takes the training history and returns a forecast for the next `horizon`
days, indexed by date. Models are evaluated on a holdout window: the last `horizon`
days of the series, which the models never see during training.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # xgboost is optional; scikit-learn's GBM is used instead
    XGBRegressor = None

MIN_TRAIN_DAYS = 60   # minimum training history after removing the holdout window
LAG_DAYS = 14         # the lag model looks back this many days


def metrics(actual, predicted) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    err = y - p
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err / y)) * 100),
    }


def _future(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def seasonal_naive(train: pd.Series, horizon: int) -> pd.Series:
    """Each future day equals the same weekday in the last week."""
    last_week = train.values[-7:]
    return pd.Series([last_week[i % 7] for i in range(horizon)], index=_future(train, horizon))


def holt_winters(train: pd.Series, horizon: int) -> pd.Series:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fit = ExponentialSmoothing(train.values, trend="add", seasonal="add", seasonal_periods=7).fit()
    return pd.Series(np.asarray(fit.forecast(horizon)), index=_future(train, horizon))


def sarima(train: pd.Series, horizon: int) -> pd.Series:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit = SARIMAX(train.values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    return pd.Series(np.asarray(fit.forecast(horizon)), index=_future(train, horizon))


def _lag_features(history: list, date: pd.Timestamp) -> list:
    return [
        history[-1],
        history[-2],
        history[-7],
        history[-14],
        float(np.mean(history[-7:])),
        date.dayofweek,
        date.month,
        int(date.dayofweek >= 5),
    ]


def gradient_boosting(train: pd.Series, horizon: int) -> pd.Series:
    """Lag-feature regression, applied recursively: each predicted day feeds the next."""
    values = [float(v) for v in train.values]
    dates = train.index
    X, y = [], []
    for i in range(LAG_DAYS, len(values)):
        X.append(_lag_features(values[:i], dates[i]))
        y.append(values[i])

    if XGBRegressor is not None:
        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0)
    else:
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(np.array(X, dtype=float), np.array(y, dtype=float))

    future = _future(train, horizon)
    history = list(values)
    out = []
    for day in future:
        pred = float(model.predict(np.array([_lag_features(history, day)], dtype=float))[0])
        out.append(pred)
        history.append(pred)
    return pd.Series(out, index=future)


MODELS = {
    "Seasonal Naive": seasonal_naive,
    "Holt-Winters": holt_winters,
    "SARIMA": sarima,
    "XGBoost / GBM (lag features)": gradient_boosting,
}


@dataclass
class ForecastRun:
    train: pd.Series
    test: pd.Series
    forecasts: dict = field(default_factory=dict)   # model name -> forecast Series
    results: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per model, best first
    errors: dict = field(default_factory=dict)      # model name -> reason it failed

    @property
    def best(self) -> str | None:
        return None if self.results.empty else str(self.results["Model"].iloc[0])


def run(series: pd.Series, horizon: int = 30, names=None) -> ForecastRun:
    """Train the selected models and score them on the last `horizon` days.

    names=None runs every model. An empty list runs none and returns empty results.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    if len(series) < horizon + MIN_TRAIN_DAYS:
        raise ValueError(f"need at least {horizon + MIN_TRAIN_DAYS} days of history for a {horizon}-day horizon")

    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    selected = list(MODELS) if names is None else [n for n in names if n in MODELS]

    forecasts, rows, errors = {}, [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels convergence chatter
        for name in selected:
            try:
                fc = MODELS[name](train, horizon)
                forecasts[name] = fc
                rows.append({"Model": name, **metrics(test.values, fc.values)})
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"

    if rows:
        results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    else:
        results = pd.DataFrame(columns=["Model", "RMSE", "MAE", "MAPE"])
    return ForecastRun(train=train, test=test, forecasts=forecasts, results=results, errors=errors)
