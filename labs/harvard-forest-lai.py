# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Can satellite indices track leaf area at Harvard Forest?
#
# **In class: 25–30 minutes.** Use real optical and radar images from the
# Prospect Hill tract, Massachusetts, and field measurements from the
# Environmental Measurement Site (EMS). Leaf area index (LAI) is leaf area
# per unit ground area. Here the response is the published site mean from
# field-instrument readings, not a satellite LAI estimate.
#
# 1. Download image chips and field records. Inspect a date and mask.
# 2. Compute indices per pixel, then summarize them for each date.
# 3. Compare correlations using 2022–2023. Fit `LAI = intercept + slope × index`.
# 4. Test on 2024 dates. Compare with predicting the training mean every time.
#
# The one observation per row is a **site-date**, not a pixel or an
# individual field plot. We have only three held-out dates, so this is a
# classroom comparison of seasonal relationships, not a validation for
# another forest or a definitive ranking of sensors.

# %%
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import display

CHIPS_URL = "https://vision.csi.miamioh.edu/data/cse620b/forestry/harvard-forest-chips.zip"
LAI_URL = "https://harvardforest1.fas.harvard.edu/data/p06/hf069/hf069-02-lai-site.csv"

# %% [markdown]
# ## 1. Retrieve the inputs
#
# The image archive contains actual Sentinel-2 L2A (RGB, near-infrared,
# shortwave infrared, and scene classification) and Sentinel-1 RTC
# (terrain-corrected radar) pixels on a shared 20 m grid. Nothing in the
# archive is a precomputed index. The scene table records full scene IDs
# and acquisition dates; images are within ten days of the field visit.
# The approximate sampling footprint covers the
# western half of a 500 m radius around the tower, where the survey transects
# run. It is not a map of exact plot boundaries.

# %%
if "FORESTRY_CHIPS_DIR" in os.environ:
    directory = Path(os.environ["FORESTRY_CHIPS_DIR"])
else:
    directory = Path("harvard-forest")
    response = requests.get(CHIPS_URL, timeout=60)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        archive.extractall(directory)

response = requests.get(LAI_URL, timeout=30)
response.raise_for_status()
field = pd.read_csv(BytesIO(response.content), parse_dates=["date"])
field = field.loc[field.site.eq("ems"), ["date", "n_plots", "all_mean"]]
field = field.rename(columns={"date": "field_date", "all_mean": "LAI"})
scene_dates = ["field_date", "optical_date", "radar_date"]
scenes = pd.read_csv(directory / "scenes.csv", parse_dates=scene_dates)
matched = scenes.merge(field, on="field_date", validate="one_to_one")
if len(matched) != len(scenes) or matched.LAI.isna().any():
    raise ValueError("A scene date is missing its EMS LAI measurement.")

footprint = np.load(directory / "footprint.npy")
display(matched[scene_dates + ["n_plots", "LAI", "common_pixels"]])

# %% [markdown]
# ## 2. Compute comparable indices
#
# Sentinel-2: red = B04, near-infrared = B08, SWIR1 = B11, and SWIR2 = B12.
# The chips are already in surface reflectance units. SWIR1 is used for
# NDMI and the reduced simple ratio (RSR); SWIR2 is used for NBR. Fix the
# RSR SWIR1 limits using the 2nd and 98th percentiles of usable **training**
# pixels and clip the adjustment to [0, 1], so every date uses the same
# scale. This is a classroom calibration, not the spruce study's limits.
#
# The regular radar series has **VV and VH**, not **HH and HV**. Thus
# `VV/VH` and `(VH − VV)/(VH + VV)` are analogues of the slide deck's
# RRVI and RRNDVI, **not the original indices**. Also calculate the
# dual-pol RVI `4 VH/(VV + VH)`. Use linear backscatter power, not dB.
# With this definition RVI equals twice the normalized radar difference
# plus two, so their Pearson correlations must be identical. With
# `tau = 1`, the deck's VH/VV RDPI is RVI divided by four. Try another
# positive value of `tau` and compare the results; the input bands stay
# the same.
#
# Keep only SCL class 4 (vegetation) and positive, finite band values
# inside the footprint. Use the same usable pixels for every index on a
# given date. The mask can vary by date.

# %%
def load_chip(field_date):
    with np.load(directory / f"{field_date:%Y-%m-%d}.npz") as file:
        bands = dict(file)
    valid = footprint & (bands["SCL"] == 4)
    for band in bands.values():
        valid &= np.isfinite(band) & (band > 0)
    return bands, valid


train_dates = matched.loc[matched.field_date.dt.year.lt(2024), "field_date"]
train_swir = []
for field_date in train_dates:
    bands, valid = load_chip(field_date)
    train_swir.append(bands["B11"][valid])
lo, hi = np.percentile(np.concatenate(train_swir), [2, 98])


def indices(bands):
    # Apply only to the common valid pixels, so denominators are positive.
    red, nir, swir1, swir2 = (bands[k] for k in ("B04", "B08", "B11", "B12"))
    vv, vh = bands["VV"], bands["VH"]
    sr = nir / red
    tau = 1.0
    return {
        "SR": sr,
        "NDVI": (nir - red) / (nir + red),
        "NDMI": (nir - swir1) / (nir + swir1),
        "RSR": sr * (1 - np.clip((swir1 - lo) / (hi - lo), 0, 1)),
        "NBR": (nir - swir2) / (nir + swir2),
        "VV/VH": vv / vh,
        "Radar ND": (vh - vv) / (vh + vv),
        "Dual-pol RVI": 4 * vh / (vv + vh),
        "RDPI": vh / (vh + tau * vv),
    }

# %% [markdown]
# Examine one chip. The RGB image includes areas outside the sampling
# footprint. Blank pixels in the two index maps are excluded from analysis.
# Change `example` to see another date. Can you spot water, breaks in
# vegetation, or radar speckle?

# %%
example = matched.iloc[len(matched) // 2].field_date
bands, valid = load_chip(example)
rgb = np.stack([bands[key] for key in ("B04", "B03", "B02")], axis=-1)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
ax[0].imshow(np.nan_to_num(np.clip(rgb * 4, 0, 1)))
ax[0].set_title(f"RGB near {example:%Y-%m-%d}")
values = indices({key: value[valid] for key, value in bands.items()})
for panel, name in zip(ax[1:], ("NDVI", "Dual-pol RVI")):
    layer = np.full(valid.shape, np.nan)
    layer[valid] = values[name]
    fig.colorbar(panel.imshow(layer, cmap="viridis"), ax=panel, shrink=0.72)
    panel.set_title(name)
for panel in ax:
    panel.set_axis_off()
plt.show()

# %% [markdown]
# ## 3. Rank indices on the training dates
#
# Use the pixel median as the summary for each site-date. Pearson's **r**
# measures the direction and strength of a linear association. Compare
# `abs(r)` to rank strength, but keep the sign to discuss direction.
# Pick an optical index and a radar index to compare.

# %%
summaries = []
for field_date in matched.field_date:
    bands, valid = load_chip(field_date)
    values = indices({key: value[valid] for key, value in bands.items()})
    summaries.append({name: np.median(pixels) for name, pixels in values.items()})

table = pd.concat([matched, pd.DataFrame(summaries)], axis=1)
train = table.loc[table.field_date.dt.year.lt(2024)]
test = table.loc[table.field_date.dt.year.eq(2024)]
index_names = list(summaries[0])
r = train[index_names].corrwith(train.LAI).sort_values(key=abs, ascending=False)
print(f"{len(train)} training and {len(test)} held-out site-dates")
display(r.rename("Training Pearson r").round(3).to_frame())

# %% [markdown]
# ## 4. Fit a spruce-style one-index model
#
# Set `predictor` from the training table. Fit on 2022–2023 only, then use
# the same equation to predict 2024 LAI. Record the slope, test RMSE (in LAI
# units), and whether it beats the training-mean baseline. Repeat with an
# index from the other sensor. Use 2024 for comparison, not for tuning the
# prediction rule.

# %%
predictor = "NDVI"  # Change this to compare an optical and a radar model.
slope, intercept = np.polyfit(train[predictor], train.LAI, 1)
predicted = intercept + slope * test[predictor]
rmse = np.sqrt(np.mean((predicted - test.LAI) ** 2))
baseline = np.sqrt(np.mean((train.LAI.mean() - test.LAI) ** 2))
print(f"Predicted LAI = {intercept:.2f} + ({slope:.2f}) × {predictor}")
print(f"2024 RMSE: {rmse:.2f}; training-mean baseline RMSE: {baseline:.2f}")

x = np.linspace(table[predictor].min(), table[predictor].max(), 100)
plt.scatter(train[predictor], train.LAI, label="2022–2023 fit data")
plt.scatter(test[predictor], test.LAI, marker="D", label="2024 held out")
plt.plot(x, intercept + slope * x, label="Training fit")
plt.xlabel(predictor)
plt.ylabel("Field-measured site mean LAI")
plt.legend()
plt.show()

# %% [markdown]
# ## Discuss
#
# - Which index was strongest in training? Did its slope and error on 2024
#   support the same interpretation? Did either model beat the baseline?
# - Why do some radar indices give the same training correlation?
# - The radar and optical scenes are not always from the field date. How
#   could a wet canopy or quick spring leaf-out change the comparison?
# - Would a relationship driven by spring and autumn work equally well
#   across stands observed on the same midsummer date? What measurements
#   would you need to test that?
#
# ## Data and worked examples
#
# - [Harvard Forest HF069: Biomass Inventories at the EMS Tower](https://harvardforest1.fas.harvard.edu/exist/apps/datasets/showData.html?id=hf069).
#   Uses the `ems` rows of the published site LAI table, not a gridded LAI product.
# - Contains modified Copernicus Sentinel data (2022–2024), retrieved through
#   [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
#   and [Sentinel-1 RTC](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc).
# - [Rasterio walk-through for Sentinel-2 NDVI](https://www.forestry-ecological-gis.org/ecological-gis-data-foundations-in-python/vegetation-index-calculation-in-python/calculating-ndvi-from-sentinel-2-with-rasterio/),
#   including surface-reflectance offsets and masked pixels.
# - [Planetary Computer Sentinel-1 RTC tutorial](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/tutorials/customizable-rtc-sentinel1.ipynb).
# - [ESA PhiLab radar index examples](https://github.com/ESA-PhiLab/sarpyx/blob/main/docs/user_guide/science_applications.md).
