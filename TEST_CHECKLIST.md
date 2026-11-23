# Dashboard test checklist

1. Confirm the app starts and loads `data/drilling_demonstration_dataset.csv.gz` automatically.
2. Confirm the sidebar reports 59,163 observations and 118 columns.
3. Review Executive Overview KPIs, ROP trend, lithology distribution and screening-score distribution.
4. Open Data Explorer; filter depth/date, select several numeric signals, inspect the statistics and download a filtered CSV.
5. Train each available ROP model; review chronological metrics, predictions and feature importance.
6. Run the pre-computed dysfunction screening review, then run Isolation Forest and download the results.
7. Train the time-series baseline and inspect the actual-versus-predicted plot.
8. Train an ROP model, then run a Digital Twin scenario within the observed data envelope.
9. In Advanced Data Science, test correlation, PCA and K-Means with selected numeric fields.
10. Ask the AI Drilling Assistant about ECD, hole cleaning, ROP, torque and stuck pipe.
11. Generate and download the Excel engineering pack.
12. Confirm the app works without API keys or an Ollama installation.

Record any Streamlit errors and screenshots. All model and screening outputs are decision-support demonstrations and require engineering validation.
