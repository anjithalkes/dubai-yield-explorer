"""
Dubai Community Rental Yield Explorer
=====================================
Investor dashboard for the MSc dissertation (UC1 map, UC3 explanation, FR6 disclaimer, NFR5 speed).

The dashboard never runs a model. It reads copies of three pipeline outputs, kept in the
application data folder PROGRAM/APP/data (DATA_DIR; copied from OUTPUT_DATA/Processed_Data):
    explain_latest_primary.parquet      Step 8: prediction, GeoShapley parts, accuracy check
    predictions_latest_primary.parquet  Step 7: community centroids
    community_boundaries.geojson        Step 2: community polygons (copy of communities_matched.geojson)
and appends one row per explanation shown to tc10_timing_log_primary.csv (TC10 evidence, LOG_DIR).

Code location: /content/drive/MyDrive/ESML_PROJECT/PROGRAM/APP/dashboard_app.py
Start it in Colab with run_dashboard_colab.py (same folder), or deploy it online on Streamlit
Community Cloud with a "data" folder next to this file.

Version 6 – click a community on the map to open it:
  * clicking a polygon selects that community exactly as choosing it in the drop-down list does
    (same zoom, name label, yield, explanation, driver chart and TC10 log row)
  * the map is drawn with streamlit-folium, which sends the clicked polygon back to the app;
    only polygon clicks cause a rerun (panning and zooming do not)
  * clicking an area without an estimate shows a short note instead
Version 5 (V45 of the dissertation) – same look as version 4, faster and more accurate:
  * data held once in memory (st.cache_resource) instead of being copied on every click
  * map colours, outlines and tooltips prepared once per community, not on every redraw
  * finished maps cached for up to 64 filter/selection combinations
  * the TC10 log gets one row per community actually opened (no duplicates when the
    slider, the checkbox or the log switch is used)
  * the Legend panel reads "≤ 4%" and "≥ 10%", because values outside 4–10% are clipped
    (the colour bar at the top keeps the short version 4 caption)
Version 4 (V44 of the dissertation):
  * green-to-red colour scale for predicted yield, fixed range 4–10% (highest prediction 9.8%)
  * the selected community's name is shown on its polygon when the map zooms to it
  * short labels DXB, DWC and CBD next to the point-of-interest markers
  * faster map: simplified and rounded polygons, canvas rendering, per-community styles
    prepared once, and the finished map HTML cached for each filter/selection combination
  * communities with too little data are left unfilled (thin grey outline)
  * driver bars use orange (raises yield) and slate grey (lowers yield)
"""
import json
import os
import time

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from branca.element import MacroElement, Template
from streamlit_folium import st_folium

# --------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------
# Colab sets DATA_DIR to the Drive stage folder; online (Streamlit Community Cloud) the data sit in
# a "data" folder next to this file.
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
LOG_DIR = os.environ.get("LOG_DIR", DATA_DIR)   # TC10 log; the launcher points this to Drive
TAG = os.environ.get("YIELD_TAG", "primary")
SNAPSHOT = "2026 Q3"
SPATIAL_MAE = 1.157            # spatial block CV MAE of the final LightGBM model (Section 6.4.4)
YIELD_MIN, YIELD_MAX = 4, 10   # fixed colour range of the map (%); highest prediction is 9.8%
YIELD_TICKS = [4, 5, 6, 7, 8, 9, 10]
GREEN_TO_RED = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027"]
NO_DATA_OUTLINE = "#8c8c8c"    # communities without an estimate: no fill, thin grey outline
RAISES, LOWERS = "#E6550D", "#6B7B8C"   # driver bars: orange raises, slate grey lowers the yield
SIMPLIFY_DEG = 0.0002          # polygon simplification tolerance (about 20 m), only for drawing
DUBAI_CENTER, DUBAI_ZOOM = (25.12, 55.25), 10
ALL = "— All of Dubai —"

LANDMARKS = [  # short label, full name, latitude, longitude, Font Awesome icon, marker colour
    ("DXB", "Dubai International Airport", 25.2532, 55.3657, "plane", "darkblue"),
    ("DWC", "Al Maktoum International Airport", 24.8960, 55.1614, "plane", "darkblue"),
    ("CBD", "Burj Khalifa, Downtown Dubai", 25.1972, 55.2744, "building", "black"),
]

LABELS = {"dist_metro_km": "distance to the metro", "dist_dwc_km": "distance to DWC airport",
          "dist_cbd_km": "distance to the CBD (Downtown)", "dist_coast_km": "distance to the coast",
          "area_km2": "community size", "share_villa": "share of villas",
          "share_freehold": "share of freehold units", "bedrooms_median": "typical number of bedrooms",
          "area_m2_median": "typical floor area", "year": "market conditions (year)",
          "qtr": "season (quarter)", "location": "location"}
FEATURES = list(LABELS)[:-1]

st.set_page_config(page_title="Dubai Rental Yield Explorer", page_icon="🏙️", layout="wide",
                   initial_sidebar_state="collapsed")

COLOURS = cm.LinearColormap(GREEN_TO_RED, vmin=YIELD_MIN, vmax=YIELD_MAX,
                            caption="Predicted gross yield (%)",
                            tick_labels=YIELD_TICKS)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def check_files():
    """Stop with a clear message if a required file is missing from the Drive folder."""
    needed = [f"explain_latest_{TAG}.parquet", f"predictions_latest_{TAG}.parquet",
              "community_boundaries.geojson"]
    missing = [f for f in needed if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        st.error(f"Cannot find {', '.join(missing)} in {DATA_DIR}. Mount Google Drive and run "
                 "run_dashboard_colab.py, which prepares these files.")
        st.stop()


def _round_coords(obj, nd=5):
    """Round nested coordinate lists to nd decimals (about 1 m), which shrinks the map HTML."""
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), nd) for v in obj]
        return [_round_coords(o, nd) for o in obj]
    return obj


def _light_boundaries(boundaries):
    """Keep only area_id, simplify each polygon for drawing and round its coordinates."""
    try:
        from shapely.geometry import mapping, shape
    except ImportError:                              # shapely missing: rounding only
        shape = None
    out = []
    for feat in boundaries["features"]:
        aid = feat["properties"].get("area_id")
        geom = feat["geometry"]
        if shape is not None and geom:
            geom = mapping(shape(geom).simplify(SIMPLIFY_DEG, preserve_topology=True))
        out.append({"type": "Feature", "properties": {"area_id": int(aid) if aid is not None else -1},
                    "geometry": {"type": geom["type"], "coordinates": _round_coords(geom["coordinates"])}})
    return {"type": "FeatureCollection", "features": out}


@st.cache_resource(show_spinner="Loading the stored predictions …")
def load_data():
    """Read the explanations, centroids and boundaries once and keep them in memory.
    cache_resource returns the same objects on every rerun (no copy), so a click does not
    re-copy about 1.5 MB of data; the page code only reads them and never changes them."""
    t0 = time.perf_counter()
    explain = pd.read_parquet(os.path.join(DATA_DIR, f"explain_latest_{TAG}.parquet"))
    centroids = pd.read_parquet(os.path.join(DATA_DIR, f"predictions_latest_{TAG}.parquet"),
                                columns=["area_id", "lat", "lon"]).drop_duplicates("area_id")
    df = explain.merge(centroids, on="area_id", how="left")
    df["area_name_en"] = [n.title() if n.isupper() else n for n in df["area_name_en"].astype(str)]
    df["area_id"] = df["area_id"].astype(int)
    parts = df[[f"gs_{f}" for f in FEATURES]].rename(columns=lambda c: c[3:])
    parts["location"] = df["location_total"]
    with open(os.path.join(DATA_DIR, "community_boundaries.geojson"), encoding="utf-8") as f:
        boundaries = _light_boundaries(json.load(f))
    styles = _community_styles(df)                     # colour, outline and tooltip per community
    return df, parts, boundaries, styles, time.perf_counter() - t0


def _community_styles(df):
    """Style properties for every community, computed once (they do not depend on the filter)."""
    styles = {}
    for r in df.itertuples():
        y = float(np.clip(r.pred_yield, YIELD_MIN, YIELD_MAX))   # clip to the 4–10% scale
        styles[int(r.area_id)] = dict(
            label=f"{r.area_name_en}: {r.pred_yield:.1f}%" + (" (outside training range)" if r.extrapolated else "")
                  + " – click for details",
            fill=COLOURS(y), fo=0.75,
            line="#212121" if r.extrapolated else "#ffffff",
            w=2 if r.extrapolated else 0.6, dash="5,4" if r.extrapolated else "")
    return styles


NO_DATA_STYLE = dict(label="Insufficient data – no estimate", fill="#ffffff", fo=0,
                     line=NO_DATA_OUTLINE, w=0.8, dash="")
HIDDEN_STYLE = dict(label="", fill="#ffffff", fo=0, line="#ffffff", w=0, dash="")


# --------------------------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------------------------
LEGEND_HTML = """
{% macro html(this, kwargs) %}
<div style="position:fixed; bottom:48px; left:12px; z-index:9999; font-family:Arial,sans-serif;">
  <button onclick="var p=document.getElementById('legend-panel');
                   p.style.display = (p.style.display==='none') ? 'block' : 'none';"
          style="background:white; border:2px solid #555; border-radius:6px; padding:5px 12px;
                 font-size:13px; font-weight:bold; cursor:pointer;">&#9776; Legend</button>
  <div id="legend-panel" style="display:none; margin-top:6px; background:white; padding:10px 12px;
       border:1px solid #999; border-radius:6px; font-size:12px; line-height:1.7; width:260px;">
    <b>Predicted gross yield</b><br>
    <div style="height:12px; background:linear-gradient(to right,#1a9850,#91cf60,#d9ef8b,#fee08b,#fc8d59,#d73027);"></div>
    <div style="display:flex; justify-content:space-between;"><span>&le; 4%</span><span>7%</span><span>&ge; 10%</span></div>
    <span style="display:inline-block;width:18px;height:11px;border:3px solid #0D47A1;"></span> Selected community<br>
    <span style="display:inline-block;width:18px;height:11px;border:2px dashed #212121;"></span> Outside the training range (less reliable)<br>
    <span style="display:inline-block;width:18px;height:11px;background:#ffffff;border:1px solid #8c8c8c;"></span> No fill – insufficient data, no estimate<br>
    <i class="fa fa-plane" style="color:#0D3B66;"></i> DXB and DWC airports<br>
    <i class="fa fa-building" style="color:#000;"></i> CBD (Burj Khalifa)
  </div>
</div>
{% endmacro %}
"""

LABEL_CSS = ("font-family:Arial,sans-serif; font-weight:bold; white-space:nowrap; "
             "background:rgba(255,255,255,0.88); border-radius:4px; padding:1px 6px; "
             "box-shadow:0 0 3px rgba(0,0,0,0.35); display:inline-block;")


def _text_label(lat, lon, text, colour="#212121", size=12, dx=0, dy=0):
    """A text label on the map (DivIcon); dx/dy shift it in pixels from the point."""
    html = (f'<div style="{LABEL_CSS} color:{colour}; font-size:{size}px; '
            f'transform:translate(calc(-50% + {dx}px), calc(-50% + {dy}px));">{text}</div>')
    return folium.Marker([lat, lon], icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, 0)),
                         interactive=False)


@st.cache_data(show_spinner=False, max_entries=64)
def map_features(low, high, show_ext, selected_id):
    """Polygons with their style for one filter/selection combination, cached (at most 64 kept).
    'click' marks the polygons that open a community when clicked (shown and with an estimate)."""
    df, _, boundaries, styles, _ = load_data()
    full_range = (low, high) == (YIELD_MIN, YIELD_MAX)       # full range also keeps values beyond 4–10%
    keep = pd.Series(True, index=df.index) if full_range else df.pred_yield.between(low, high)
    keep &= show_ext | ~df.extrapolated
    if selected_id is not None:
        keep |= df.area_id == selected_id                    # the chosen community is always shown
    shown_ids = set(df.loc[keep, "area_id"])

    # Each community's style was prepared once in load_data(); here it is only looked up.
    # Only the small "properties" part is copied – the polygon coordinates are shared, not copied.
    features = []
    for feat in boundaries["features"]:
        aid = feat["properties"]["area_id"]
        style = styles[aid] if aid in shown_ids else HIDDEN_STYLE if aid in styles else NO_DATA_STYLE
        click = 1 if aid in shown_ids else 0 if aid in styles else -1   # 1 open, 0 hidden, -1 no estimate
        features.append({"type": "Feature", "properties": {"area_id": aid, "click": click, **style},
                         "geometry": feat["geometry"]})
    return {"type": "FeatureCollection", "features": features}, int(keep.sum())


def build_map(low, high, show_ext, selected_id):
    """Folium map for st_folium. The polygons come from the cache; the map object itself is light
    and is built on each run, because st_folium changes the object while drawing it."""
    df, _, _, _, _ = load_data()
    gj, n_shown = map_features(low, high, show_ext, selected_id)

    fmap = folium.Map(location=DUBAI_CENTER, zoom_start=DUBAI_ZOOM, control_scale=False, prefer_canvas=True,
                      tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                      attr="Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors")
    folium.GeoJson(gj, name="Communities", smooth_factor=1.5,
                   style_function=lambda f: {"fillColor": f["properties"]["fill"],
                                             "fillOpacity": f["properties"]["fo"],
                                             "color": f["properties"]["line"],
                                             "weight": f["properties"]["w"],
                                             "dashArray": f["properties"]["dash"] or None},
                   highlight_function=lambda f: {"weight": 3, "color": "#0D47A1"} if f["properties"]["click"] == 1 else {},
                   tooltip=folium.GeoJsonTooltip(fields=["label"], labels=False)).add_to(fmap)

    if selected_id is not None:                              # blue outline, zoom and name label
        sel = {"type": "FeatureCollection",
               "features": [f for f in gj["features"] if f["properties"]["area_id"] == selected_id]}
        layer = folium.GeoJson(sel, name="Selected", interactive=False,   # outline only, clicks pass through
                               style_function=lambda f: {"fillOpacity": 0, "color": "#0D47A1",
                                                         "weight": 5}).add_to(fmap)
        fmap.fit_bounds(layer.get_bounds(), padding=(40, 40), max_zoom=14)
        r = df.loc[df.area_id == selected_id].iloc[0]
        if pd.notna(r.lat) and pd.notna(r.lon):
            _text_label(r.lat, r.lon, r.area_name_en, colour="#0D47A1", size=14).add_to(fmap)

    for short, name, lat, lon, icon, colour in LANDMARKS:   # markers with short labels, on top
        folium.Marker([lat, lon], tooltip=f"{short} – {name}",
                      icon=folium.Icon(icon=icon, prefix="fa", color=colour)).add_to(fmap)
        _text_label(lat, lon, short, size=12, dx=30, dy=-22).add_to(fmap)

    COLOURS.add_to(fmap)
    scale = MacroElement()                                   # metric scale bar only (km)
    scale._template = Template("{% macro script(this, kwargs) %}L.control.scale({imperial: false})"
                               ".addTo({{ this._parent.get_name() }});{% endmacro %}")
    fmap.add_child(scale)
    legend = MacroElement(); legend._template = Template(LEGEND_HTML)
    fmap.get_root().add_child(legend)
    return fmap, n_shown


# --------------------------------------------------------------------------------------------
# Explanation text (formatter, Listing B.3)
# --------------------------------------------------------------------------------------------
def explanation_sentences(row, parts_row):
    """Short, contrastive sentences built from the stored GeoShapley values (Miller, 2019)."""
    lines = [f"**{row.pred_yield:.1f}%** predicted gross yield, against a city-wide model average "
             f"of {row.base_value:.1f}%."]
    for f in parts_row.abs().sort_values(ascending=False).index[:3]:
        v, label = parts_row[f], LABELS[f][0].upper() + LABELS[f][1:]
        lines.append(f"{label} has almost no effect." if abs(v) < 0.05 else
                     f"{label} {'raises' if v > 0 else 'lowers'} the yield by {abs(v):.1f} percentage points.")
    return lines


def drivers_chart(parts_row):
    """Bar chart of the six largest property and market effects (location shown separately)."""
    top = parts_row.drop("location").abs().sort_values(ascending=False).index[:6]
    bars = parts_row[top][::-1]
    fig = go.Figure(go.Bar(x=bars.values, y=[LABELS[f] for f in bars.index], orientation="h",
                           marker_color=[RAISES if v > 0 else LOWERS for v in bars.values]))
    fig.update_layout(height=280, margin=dict(l=0, r=10, t=30, b=0),
                      title="Main property and market drivers (pp)",
                      xaxis_title="Effect on predicted yield (percentage points)")
    return fig


def log_timing(community, load_s, panel_ms):
    """Append one row to the TC10 timing log (LOG_DIR, on Google Drive when run from Colab)."""
    path = os.path.join(LOG_DIR, f"tc10_timing_log_{TAG}.csv")
    try:
        pd.DataFrame([{"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                       "community": community, "data_load_ms": round(load_s * 1000, 1),
                       "panel_ms": round(panel_ms, 1)}]).to_csv(
            path, mode="a", header=not os.path.exists(path), index=False)
    except OSError:
        pass
    return path


# --------------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------------
check_files()
df, parts, boundaries, styles, load_s = load_data()
COMMUNITIES = [ALL] + sorted(df.area_name_en.unique())   # selector list
NAME_BY_ID = dict(zip(df.area_id.astype(int), df.area_name_en))   # area_id -> community name
if "community" not in st.session_state:
    st.session_state["community"] = ALL
# A polygon clicked on the map in the previous run is applied here, before the selector is drawn,
# so the selector, the map and the explanation panel all show the clicked community.
if st.session_state.get("pending_community"):
    st.session_state["community"] = st.session_state.pop("pending_community")

st.title("🏙️ Dubai Community Rental Yield Explorer")
st.caption(f"Predicted gross rental yield for {len(df)} Dubai communities, {SNAPSHOT}. "
           "Click a community on the map, or choose it from the list, to see why it received its estimate.")
st.warning("**Indicative estimates only – not investment advice.** Yields are model predictions from "
           "public EJARI and DLD records and can be wrong by about ±1.2 percentage points; estimates for communities "
           "outside the training range are less reliable. They are not official DLD figures.")

left, right = st.columns([3, 2], gap="large")

with right:                                          # community selector and reset button
    col_sel, col_btn = st.columns([4, 1.4], vertical_alignment="bottom")
    choice = col_sel.selectbox("Community", COMMUNITIES, key="community")
    col_btn.button("Show all Dubai", disabled=(choice == ALL), width="stretch",
                   on_click=lambda: st.session_state.update(community=ALL))
selected = None if choice == ALL else df.index[df.area_name_en == choice][0]

with left:                                           # UC1: filter and map
    f1, f2 = st.columns([3, 1.3], vertical_alignment="bottom")
    low, high = f1.slider("Filter by predicted yield (%)", float(YIELD_MIN), float(YIELD_MAX),
                          (float(YIELD_MIN), float(YIELD_MAX)), 0.1)
    show_ext = f2.checkbox("Include communities outside the training range", value=True)
    sel_id = None if selected is None else int(df.loc[selected, "area_id"])
    fmap, n_shown = build_map(float(low), float(high), bool(show_ext), sel_id)
    # Only polygon clicks are sent back (panning and zooming do not rerun the app).
    out = st_folium(fmap, key="map", height=640, use_container_width=True,
                    returned_objects=["last_active_drawing"])
    clicked = (out or {}).get("last_active_drawing") or {}
    props = clicked.get("properties") or {}
    note = ""
    if props.get("click") == 1:                       # a shown community with an estimate
        name = NAME_BY_ID.get(int(props["area_id"]))
        if name and name != choice:                   # open it, exactly like the drop-down list
            st.session_state["pending_community"] = name
            st.rerun()
    elif props.get("click") == -1:
        note = " The area you clicked has too few leases or sales for an estimate."
    st.caption(f"{n_shown} of {len(df)} communities shown. Click a community to open it. Unfilled areas have "
               "too few leases or sales for an estimate; dashed outlines mark communities outside the "
               "training range." + note)

with right:                                          # UC3: explanation panel or city overview
    panel_ms = None
    if selected is None:
        st.subheader("All of Dubai")
        a, b = st.columns(2)
        a.metric("Median predicted yield", f"{df.pred_yield.median():.1f}%")
        b.metric("Communities with an estimate", f"{len(df)}")
        st.markdown("**Highest predicted yields**")
        top5 = df.nlargest(5, "pred_yield")[["area_name_en", "pred_yield"]].round(1)
        st.dataframe(top5.rename(columns={"area_name_en": "Community", "pred_yield": "Yield (%)"}),
                     hide_index=True, width="stretch")
        st.info("Click a community on the map, or choose it above, to zoom to it and see what drives its estimate.")
    else:
        t0 = time.perf_counter()
        row, parts_row = df.loc[selected], parts.loc[selected]
        st.subheader(choice)
        c1, c2 = st.columns(2)
        c1.metric("Predicted gross yield", f"{row.pred_yield:.1f}%")
        c2.metric("Typical error", f"± {SPATIAL_MAE:.1f} pp")
        if row.extrapolated:
            st.info("This community lies outside the range of the training data, so its estimate is less reliable "
                    "than the typical error suggests.")
        if not bool(row.explanation_ok):
            st.error("The explanation for this community failed the accuracy check and is withheld (UC3, A2).")
        else:
            for line in explanation_sentences(row, parts_row):
                st.markdown("- " + line)
            st.plotly_chart(drivers_chart(parts_row), width="stretch")
            st.markdown("##### Location effect")
            st.markdown(f"Location {'adds' if row.location_total > 0 else 'subtracts'} "
                        f"**{abs(row.location_total):.1f} pp** (on its own {row.gs_location:+.1f} pp; "
                        f"through its interaction with the property mix {row.gs_location_x_features:+.1f} pp). "
                        "This is the part of the estimate explained by where the community is, "
                        "not by its properties.")
        panel_ms = (time.perf_counter() - t0) * 1000

# TC10: log a community once when it is opened, not again when the slider, the checkbox or the
# log switch causes a rerun (that would repeat the same row and inflate the count).
log_path = os.path.join(LOG_DIR, f"tc10_timing_log_{TAG}.csv")
if panel_ms is not None and st.session_state.get("last_logged") != choice:
    log_path = log_timing(choice, load_s, panel_ms)
    st.session_state["last_logged"] = choice
elif panel_ms is None:
    st.session_state["last_logged"] = None

st.divider()                                         # TC10: timings shown only on request
if st.toggle("Show technical log (TC10 response times)", value=False):
    m1, m2, m3 = st.columns(3)
    m1.metric("Data load (first run, then cached)", f"{load_s * 1000:.0f} ms")
    m2.metric("This explanation built in", "–" if panel_ms is None else f"{panel_ms:.0f} ms")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        m3.metric("Logged explanations", f"{len(log)}",
                  f"mean {log.panel_ms.mean():.0f} ms · max {log.panel_ms.max():.0f} ms", delta_color="off")
        st.dataframe(log.tail(10), hide_index=True, width="stretch")
    if selected is not None:
        st.caption(f"Local accuracy error for {choice}: {df.loc[selected].local_accuracy_error:.6f} pp. "
                   f"NFR5 target: an explanation appears in under 3 seconds. Log file: {log_path}")
