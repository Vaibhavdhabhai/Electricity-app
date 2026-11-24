"""Quick check without the UI. Run from the project folder:  python test_pipeline.py

Prints the series length, the model comparison table, the anomaly counts and the generated advice.
The full automated tests are in the tests/ folder.
"""
import agent
import anomaly
import data
import models

s = data.load_series(data.ALL_INDIA)
print(len(s), "days,", s.index.min().date(), "to", s.index.max().date())

run = models.run(s, 30)
print(run.results.round(2).to_string(index=False))
if run.errors:
    print("skipped:", run.errors)

an = anomaly.detect(s)
print("anomalies", int(an.anomaly.sum()), "peaks", int(an.peak.sum()))

best_metrics = run.results.iloc[0].drop("Model").astype(float).to_dict()
ctx = agent.build_context(data.ALL_INDIA, s, run.forecasts[run.best], run.best, best_metrics, an)
advice = agent.advise(ctx)
print(f"\nInsights source: {advice['source']}" + (f" ({advice['note']})" if advice["note"] else ""))
print("Why demand changes:")
for item in advice["explanation"]:
    print("  -", item)
print("Recommended actions:")
for item in advice["recommendations"]:
    print("  *", item)
