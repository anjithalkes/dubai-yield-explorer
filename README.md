# Dubai Community Rental Yield Explorer

Investor dashboard for the MSc Artificial Intelligence dissertation of Anjitha Senarath (24077444), London Metropolitan University.

Predicted gross rental yield for 144 Dubai communities (2026 Q3) on a green-to-red scale from 4% to 10%. Click a community on the map, or choose it from the list, to see why it received its estimate.

**Indicative estimates only – not investment advice.** The predictions come from public EJARI and DLD records and can be wrong by about ±1.2 percentage points.

| File | Content |
|---|---|
| `dashboard_app.py` | Streamlit app (reads stored results only; never runs a model) |
| `data/explain_latest_primary.parquet` | predictions and GeoShapley explanations (pipeline Step 8) |
| `data/predictions_latest_primary.parquet` | community centroids (pipeline Step 7) |
| `data/community_boundaries.geojson` | community polygons (pipeline Step 2) |
