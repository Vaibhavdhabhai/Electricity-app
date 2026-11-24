"""Load and clean the POSOCO daily state-wise electricity demand data.

Two files ship in data/:
  dataset_tk.csv  - wide format: one row per date, one column per state (Mega Units, MU)
  long_data_.csv  - one row per state with its coordinates (used for the weather lookup)
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
WIDE_FILE = DATA_DIR / "dataset_tk.csv"
LONG_FILE = DATA_DIR / "long_data_.csv"
# Optional offline weather for All India: an Open-Meteo CSV export saved here.
# If present, it is used instead of calling the API.
WEATHER_FILE = DATA_DIR / "weather.csv"

ALL_INDIA = "All India"
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
# After this date the dataset has only about 5 observed days per month, so it is excluded.
DENSE_END = "2020-05-31"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


class WeatherError(RuntimeError):
    """Raised when weather data cannot be fetched (no internet, API change, etc.)."""


@lru_cache(maxsize=1)
def _raw() -> pd.DataFrame:
    d = pd.read_csv(WIDE_FILE)
    d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"], format=DATE_FORMAT)
    return d


@lru_cache(maxsize=1)
def _wide() -> pd.DataFrame:
    """One row per date. Duplicate dates are averaged."""
    return _raw().groupby("Date").mean(numeric_only=True).sort_index()


def _state_series(entity: str) -> pd.Series:
    wide = _wide()
    if entity == ALL_INDIA:
        return wide.sum(axis=1)
    if entity not in wide.columns:
        raise KeyError(f"Unknown region: {entity!r}")
    return wide[entity]


def entities() -> list[str]:
    return [ALL_INDIA] + list(_wide().columns)


def load_series(entity: str = ALL_INDIA, dense_only: bool = True) -> pd.Series:
    """Daily demand (MU) with no gaps. Missing days are filled by linear interpolation."""
    s = _state_series(entity)
    if dense_only:
        s = s.loc[:DENSE_END]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    s = s.interpolate(limit_direction="both")
    s.name = "MU"
    return s


def data_quality(entity: str = ALL_INDIA) -> dict:
    """Plain-language facts about how much of the series is real observation."""
    raw_dates = _raw()["Date"]
    observed = _state_series(entity).loc[:DENSE_END]
    expected = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    return {
        "dense_period": f"{expected[0].date()} to {expected[-1].date()}",
        "observed_days": int(len(observed)),
        "interpolated_days": int(len(expected) - len(observed)),
        "extra_rows_on_duplicate_dates_averaged": int(raw_dates.duplicated().sum()),
    }


def coords(entity: str = ALL_INDIA) -> tuple[float, float]:
    loc = pd.read_csv(LONG_FILE).drop_duplicates("States").set_index("States")
    if entity == ALL_INDIA:
        return float(loc.latitude.mean()), float(loc.longitude.mean())
    return float(loc.loc[entity, "latitude"]), float(loc.loc[entity, "longitude"])


def read_weather_file(path=WEATHER_FILE) -> pd.DataFrame:
    """Read an Open-Meteo CSV export. Metadata lines before the 'time,' header are skipped."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        header = next((i for i, line in enumerate(lines) if line.lower().startswith("time,")), None)
        if header is None:
            raise WeatherError("weather file has no 'time' header row")
        df = pd.read_csv(io.StringIO("\n".join(lines[header:])))
        temp_col = next((c for c in df.columns if c.startswith("temperature_2m")), None)
        hum_col = next((c for c in df.columns if c.startswith("relative_humidity_2m")), None)
        if temp_col is None:
            raise WeatherError("weather file has no temperature column")
        wx = pd.DataFrame(
            {"temp": df[temp_col].values,
             "humidity": df[hum_col].values if hum_col else np.nan},
            index=pd.to_datetime(df["time"]),
        )
    except (OSError, ValueError, KeyError) as exc:
        raise WeatherError(f"could not read weather file ({type(exc).__name__})") from exc
    return wx.interpolate(limit_direction="both")


def get_weather(entity: str, start, end) -> pd.DataFrame:
    """Daily mean temperature and humidity.

    For All India, uses data/weather.csv if it exists. Otherwise fetches from Open-Meteo
    (free, needs internet). Raises WeatherError on failure so the caller can show a message.
    For the API, All India uses the average of the state coordinates (a rough centre).
    """
    if entity == ALL_INDIA and WEATHER_FILE.exists():
        wx = read_weather_file(WEATHER_FILE)
        return wx.loc[pd.Timestamp(start):pd.Timestamp(end)]

    lat, lon = coords(entity)
    params = dict(
        latitude=lat,
        longitude=lon,
        start_date=pd.Timestamp(start).date().isoformat(),
        end_date=pd.Timestamp(end).date().isoformat(),
        daily="temperature_2m_mean,relative_humidity_2m_mean",
        timezone="Asia/Kolkata",
    )
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        wx = pd.DataFrame(
            {"temp": daily["temperature_2m_mean"], "humidity": daily["relative_humidity_2m_mean"]},
            index=pd.to_datetime(daily["time"]),
        )
    except (OSError, requests.RequestException, KeyError, ValueError) as exc:
        raise WeatherError(f"could not fetch weather ({type(exc).__name__})") from exc
    return wx.interpolate(limit_direction="both")
