# AI Electricity Demand Forecasting & Energy Optimization

Predict → Analyze → Explain → Recommend.

A Streamlit dashboard that forecasts daily electricity demand for India and its states,
flags unusual days, and explains the patterns in plain language. Data: POSOCO daily
state-wise consumption (Kaggle), 2019–2020, in Mega Units (MU) per day.

## Quick start

Requires Python 3.10 or newer. Tested with Python 3.13, pandas 3.1, scikit-learn 1.9,
statsmodels 0.15, XGBoost 3.4 and Streamlit 1.67.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501.

### Optional settings

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Free key from Google AI Studio. Enables free-form answers in the **Ask the AI** chat. `GOOGLE_API_KEY` also works. |
| `GEMINI_MODEL` | Overrides the Gemini model (default `gemini-2.5-flash`). |
| `GROQ_API_KEY` | Free key from console.groq.com (no card needed). Used for the chat if no Gemini key is set. |
| `GROQ_MODEL` | Overrides the Groq model (default `llama-3.3-70b-versatile`; check Groq's model list). |
| `ANTHROPIC_API_KEY` | Enables Claude-written insights, and the chat if no Gemini or Groq key is set. Without it, rule-based text is shown. |
| `ANTHROPIC_MODEL` | Overrides the Claude model (default `claude-sonnet-5-5`). |

The chat uses Gemini first, then Groq, then Claude, then built-in rules. Insights use Claude when its key is
set, and rule-based text otherwise. Both fall back to rules if the API call fails, and say why.

Example: `export ANTHROPIC_API_KEY=sk-ant-...` before running Streamlit.

The **Temperature** checkbox in the sidebar compares demand with daily temperature. For
All India it uses `data/weather.csv` if that file exists. Otherwise it fetches from
Open-Meteo (free, no key, needs internet). If neither works, the app shows a warning and
carries on.

To use an offline weather file, open this URL in a browser and save the result as
`data/weather.csv`:

```
https://archive-api.open-meteo.com/v1/archive?latitude=23.18&longitude=81.79&start_date=2019-01-02&end_date=2020-05-23&daily=temperature_2m_mean,relative_humidity_2m_mean&timezone=Asia/Kolkata&format=csv
```

The coordinates (23.18, 81.79) are the average of the state coordinates, the same centre the app
uses for All India. The file must have a `time,` header row and a `temperature_2m...` column.
One point cannot represent all of India, so treat the result as a rough proxy.

## Running the tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

The tests cover data loading, all four models, anomaly detection, the Claude fallback
paths, the weather failure path, and the Streamlit app itself. They run headlessly and
do not need an API key or internet.

## Project layout

```
test_pipeline.py  Quick check without the UI (prints the model table and advice)
app.py            Streamlit dashboard (five tabs: Overview, Forecast, Peaks & Anomalies, AI Insights, Ask the AI)
chatbot.py        Chat assistant: Groq, then Claude, then rule-based answers; uses only the dashboard's facts
data.py           Loading, cleaning, data-quality facts, and the optional weather lookup
models.py         Seasonal Naive, Holt-Winters, SARIMA, and a lag-feature GBM/XGBoost model; RMSE/MAE/MAPE
anomaly.py        Robust z-score and Isolation Forest anomalies; top-5% peak days
agent.py          Builds the facts summary; Claude or rule-based explanations and recommendations
data/             dataset_tk.csv (wide: one column per state), long_data_.csv (state coordinates)
tests/            Unit tests (test_units.py), offline weather and Claude tests (test_offline_paths.py),
                  chat tests (test_chatbot.py), headless app tests (test_app.py)
requirements.txt  Dependencies
```

## How it works

1. **Clean.** Duplicate dates are averaged. Missing days are filled by linear
   interpolation. Data after May 2020 is excluded, because only about five days per month
   are recorded after that point. The Overview tab reports how many days are interpolated.
2. **Forecast.** The last `horizon` days are held out as a test window. Each model is
   trained on the rest and scored on that window by RMSE, MAE and MAPE. The best model by
   RMSE is used for the insights.
3. **Detect.** A day is an anomaly if its robust z-score against its local weekly level is
   above the threshold, or if an Isolation Forest flags it. Peaks are the top 5% of days.
4. **Explain.** Facts are summarised as JSON. Claude turns them into explanations and
   recommendations. Claude is told to use only those facts. If the call fails, the app
   shows rule-based text and says why.

## Known limits

- **Interpolated days are not observations.** Gaps are filled, so treat patterns that depend
  on them with care.
- **Duplicate dates are averaged.** The raw file has a few dates with two rows each. Averaging
  is a simplification. The Overview tab reports how many extra rows were averaged.
- **Forecasts do not use weather.** Future temperatures are unknown, so the models use only
  past demand and the calendar. Temperature is used only to show its correlation with demand.
  That correlation also contains seasonal effects, so it shows association, not cause.
- **Short history.** About 16 months of usable data is enough to show weekly patterns, but
  not yearly ones.
- **Phase note.** If this is used for an academic phase 1 synopsis, do not present model
  results. The README originally said the same.
