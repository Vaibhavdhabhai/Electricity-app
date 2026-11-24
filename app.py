"""Streamlit dashboard. Run with:  streamlit run app.py"""
import plotly.graph_objects as go
import streamlit as st

import agent
import anomaly
import chatbot
import data
import models

st.set_page_config(page_title="Electricity Demand AI", layout="wide")
st.title("⚡ AI Electricity Demand Forecasting & Energy Optimization")
st.caption("Predict → Analyze → Explain → Recommend · Data: POSOCO via Kaggle (India, 2019–2020)")


# ---- cached computations (each depends only on its arguments) ----------------

@st.cache_data(show_spinner=False)
def get_series(entity):
    return data.load_series(entity)


@st.cache_data(show_spinner="Training forecasting models…")
def get_forecasts(entity, horizon, model_names):
    return models.run(get_series(entity), horizon, list(model_names))


@st.cache_data(show_spinner=False)
def get_anomalies(entity, z_thr):
    return anomaly.detect(get_series(entity), z_thr=z_thr)


@st.cache_data(show_spinner="Fetching temperature data…")
def get_temperature_corr(entity):
    """Returns (correlation, error_message). Exactly one of the two is None."""
    s = get_series(entity)
    try:
        wx = data.get_weather(entity, s.index.min(), s.index.max())
    except data.WeatherError as exc:
        return None, str(exc)
    return float(s.corr(wx["temp"].reindex(s.index))), None


# ---- sidebar -----------------------------------------------------------------

with st.sidebar:
    entity = st.selectbox("Region / State", data.entities())
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    model_names = st.multiselect("Models", list(models.MODELS), default=list(models.MODELS))
    z_thr = st.slider("Anomaly z-score threshold", 2.0, 6.0, 3.5, 0.5)
    use_weather = st.checkbox("Compare with temperature (uses data/weather.csv or Open-Meteo)")

series = get_series(entity)
forecast_run = get_forecasts(entity, horizon, tuple(model_names))
an = get_anomalies(entity, z_thr)

# Facts shared by the AI Insights and chat tabs. Built once per rerun.
ctx = None
weather_error = None
if forecast_run.best is not None:
    best = forecast_run.best
    metrics_row = forecast_run.results.iloc[0].drop("Model").astype(float).round(2).to_dict()
    corr, weather_error = get_temperature_corr(entity) if use_weather else (None, None)
    ctx = agent.build_context(
        entity=entity,
        series=series,
        fc_series=forecast_run.forecasts[best],
        best=best,
        metrics=metrics_row,
        anomalies=an,
        corr_temp=corr,
    )

tab_overview, tab_forecast, tab_peaks, tab_ai, tab_chat = st.tabs(
    ["Overview", "Forecast", "Peaks & Anomalies", "AI Insights", "Ask the AI"]
)

# ---- Overview ----------------------------------------------------------------

with tab_overview:
    c = st.columns(4)
    c[0].metric("Days", len(series))
    c[1].metric("Mean (MU/day)", f"{series.mean():,.1f}")
    c[2].metric("Peak (MU)", f"{series.max():,.1f}")
    c[3].metric("Anomalies", int(an["anomaly"].sum()))

    fig = go.Figure(go.Scatter(x=series.index, y=series, name="Demand"))
    fig.add_scatter(x=series.index, y=series.rolling(14).mean(), name="14-day mean")
    fig.update_layout(yaxis_title="MU", height=380)
    st.plotly_chart(fig)

    weekday = series.groupby(series.index.day_of_week).mean()
    weekday.index = agent.DAYS
    st.bar_chart(weekday)
    st.caption("Average demand by weekday. Interpolated days are not real observations.")

    with st.expander("Data quality"):
        q = data.data_quality(entity)
        st.table({k.replace("_", " ").capitalize(): [v] for k, v in q.items()})

# ---- Forecast ----------------------------------------------------------------

with tab_forecast:
    if not model_names:
        st.info("Select at least one model in the sidebar.")
    elif forecast_run.results.empty:
        st.error("No model produced a forecast.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=forecast_run.train.index[-90:], y=forecast_run.train.values[-90:], name="Train (last 90 days)")
        fig.add_scatter(x=forecast_run.test.index, y=forecast_run.test.values, name="Actual",
                        line=dict(color="black"))
        for name, fc in forecast_run.forecasts.items():
            fig.add_scatter(x=fc.index, y=fc.values, name=name, line=dict(dash="dash"))
        fig.update_layout(yaxis_title="MU", height=420)
        st.plotly_chart(fig)

        st.dataframe(
            forecast_run.results.style.format({"RMSE": "{:.1f}", "MAE": "{:.1f}", "MAPE": "{:.2f}"}),
            hide_index=True,
        )
        st.success(f"Best model by RMSE on the last {horizon} days: {forecast_run.best}")

    for name, msg in forecast_run.errors.items():
        st.warning(f"{name} skipped: {msg}")

# ---- Peaks & anomalies -------------------------------------------------------

with tab_peaks:
    fig = go.Figure(go.Scatter(x=series.index, y=series, name="Demand", line=dict(color="#1BA5A0")))
    peaks, flagged = an[an["peak"]], an[an["anomaly"]]
    fig.add_scatter(x=peaks.index, y=peaks["value"], mode="markers", name="Peak (top 5%)",
                    marker=dict(color="orange", size=7))
    fig.add_scatter(x=flagged.index, y=flagged["value"], mode="markers", name="Anomaly",
                    marker=dict(color="red", size=9, symbol="x"))
    fig.update_layout(yaxis_title="MU", height=420)
    st.plotly_chart(fig)

    st.dataframe(flagged[["value", "z", "zscore_flag", "iso_flag"]].round(2))

# ---- AI insights -------------------------------------------------------------

with tab_ai:
    if ctx is None:
        st.info("Run at least one forecasting model to generate insights.")
    else:
        if use_weather and weather_error:
            st.warning(f"Weather unavailable ({weather_error}). Insights exclude temperature.")

        out = agent.advise(ctx)
        caption = f"Insights source: {out['source']}"
        if out["note"]:
            caption += f" · {out['note']}"
        st.caption(caption)

        st.subheader("Why is demand changing?")
        for item in out["explanation"]:
            st.markdown(f"- {item}")

        st.subheader("Recommended actions")
        for item in out["recommendations"]:
            st.markdown(f"- {item}")

        with st.expander("Facts given to the agent"):
            st.json(ctx)

# ---- Ask the AI --------------------------------------------------------------

with tab_chat:
    if ctx is None:
        st.info("Run at least one forecasting model to use the chat.")
    else:
        st.caption("Ask about this region. Answers use only the figures on this dashboard.")
        if st.session_state.get("chat_entity") != entity:
            st.session_state.chat_entity = entity
            st.session_state.chat_messages = []
        messages = st.session_state.chat_messages

        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("source"):
                    st.caption(f"Answered by: {m['source']}")

        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input("Your question", placeholder="e.g. Which day had the highest demand?")
            submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            history = [{"role": m["role"], "content": m["content"]} for m in messages]
            res = chatbot.answer(question, ctx, history)
            text = res["answer"] + (f"\n\n*{res['note']}*" if res["note"] else "")
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": text, "source": res["source"]})
            st.rerun()
