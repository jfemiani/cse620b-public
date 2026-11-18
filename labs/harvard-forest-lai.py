# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Can satellites tell how leafy a forest is?
#
# **In class · 25–30 minutes**
#
# Look at the same part of Harvard Forest in June and November. Where do you expect the biggest change in leaf cover? Today you'll compare two ways of seeing that change from space: Sentinel-2 measures reflected light, and Sentinel-1 measures the strength of a returning radar signal.
#
# Run the cells in order. You'll look at the images, compare satellite indices with measurements taken in the forest, and test whether you can predict how much leaf area was there on three dates in 2024. At the end, be ready to show which index you tried and what happened on those three dates.
#
# ## 1. Look at the same forest in two seasons
#
# Run the setup cell to load the satellite images and field measurements. The next cell shows two color images of the Prospect Hill tract in Massachusetts. Both panels cover the same area and use the same display settings.

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

# %%
example_dates = ["2023-06-05", "2023-11-07"]  # Field visit dates
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for ax, visit in zip(axes, example_dates):
    field_date = pd.Timestamp(visit)
    scene = matched.loc[matched.field_date.eq(field_date)].iloc[0]
    with np.load(directory / f"{field_date:%Y-%m-%d}.npz") as chip:
        rgb = np.stack([chip[band] for band in ("B04", "B03", "B02")], axis=-1)
    ax.imshow(np.nan_to_num(np.clip(rgb * 4, 0, 1)))
    ax.contour(footprint, levels=[0.5], colors="white", linewidths=1)
    ax.set_title(f"Sentinel-2 · {scene.optical_date:%b %d, %Y}")
    ax.set_axis_off()
plt.show()

# %% [markdown]
# **With your group:** Look inside the white outline. Where did the color change most between June and November? Would you expect optical imagery and radar to respond to the same changes?
#
# ## 2. Pair the images with measurements on the ground
#
# Researchers at the Environmental Measurement Site (EMS) measured **leaf area index (LAI)**, the amount of leaf area per unit of ground area. The table gives the published mean across the plots measured on each visit. The white outline approximates the area covered by the field survey transects. It is roughly the western half of a 500 m radius around the tower.
#
# Each row pairs one field visit with a nearby optical image and radar image. Both images are on the same 20 m grid and were acquired within ten days of the field visit.
#
# **Check:** Find the June and November visits in the table. Does the difference in measured LAI agree with what you saw in the color images? How far apart were the field visit and each satellite acquisition?

# %%
display(matched[scene_dates + ["n_plots", "LAI"]])


# %% [markdown]
# ## 3. Turn satellite pixels into numbers you can compare with LAI
#
# An **index** combines bands into a number for each pixel. For example, NDVI compares near-infrared and red reflectance. The dual-polarization radar vegetation index (RVI) uses the two radar channels, VV and VH. The code below defines both indices and several alternatives for your group to try.
#
# Within the white outline, keep the pixels that Sentinel-2 classifies as vegetation and that have valid measurements in all bands. This gives the optical and radar indices the same set of pixels to work with on each date. The color image shows the full square scene. The index maps cover the same area, but the code leaves pixels outside the white outline blank. Gaps inside the outline are pixels that failed the vegetation or validity checks. Try changing `example` to `example_dates[1]` to see what changes in November.

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


# %%
example = pd.Timestamp(example_dates[0])
bands, valid = load_chip(example)
rgb = np.stack([bands[key] for key in ("B04", "B03", "B02")], axis=-1)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
ax[0].imshow(np.nan_to_num(np.clip(rgb * 4, 0, 1)))
ax[0].contour(footprint, levels=[0.5], colors="white", linewidths=1)
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
# You now have many satellite pixels for each field visit and one field LAI value. Take the **median of each index** across the usable pixels. The resulting table has one row per field visit with measured LAI and one summary value per index.

# %%
summaries = []
for field_date in matched.field_date:
    bands, valid = load_chip(field_date)
    values = indices({key: value[valid] for key, value in bands.items()})
    summaries.append({name: np.median(pixels) for name, pixels in values.items()})

table = pd.concat([matched, pd.DataFrame(summaries)], axis=1)

display(table[["field_date", "LAI", "NDVI", "Dual-pol RVI"]].round({"LAI": 3, "NDVI": 3, "Dual-pol RVI": 3}))

# %% [markdown]
# ## 4. Pick an optical index and a radar index
#
# Use the **2022–2023 visits** to see which indices change most consistently with LAI. Pearson's correlation, *r*, measures how closely the points follow a straight line. Values near −1 and +1 both indicate a strong relationship. The sign tells you whether the index tends to fall or rise as LAI increases.
#
# Start with **NDVI** and **Dual-pol RVI**. Find them in the table below and predict which will do better on the 2024 visits. Your group can also use the table to choose a different optical or radar index before looking at the 2024 predictions.

# %%
train = table.loc[table.field_date.dt.year.lt(2024)]
test = table.loc[table.field_date.dt.year.eq(2024)]
index_names = list(summaries[0])
r = train[index_names].corrwith(train.LAI).sort_values(key=abs, ascending=False)
print(f"{len(train)} training and {len(test)} held-out site-dates")
display(r.rename("Training Pearson r").round(3).to_frame())

# %% [markdown]
# ## 5. Predict LAI for the 2024 visits
#
# For each index, fit a line relating the index to measured LAI using the 2022–2023 visits. Then use that line and the 2024 satellite index values to predict LAI for the three 2024 visits. The diamond markers show what was actually measured on those dates.
#
# Compare each model's **RMSE** (root mean squared error, in LAI units) with a simple baseline that predicts the 2022–2023 mean LAI every time. Lower error is better. If your group chose different indices in step 4, enter their names before running this cell.

# %%
optical_index = "NDVI"
radar_index = "Dual-pol RVI"

train_mean = train.LAI.mean()
predictions = test[["field_date", "LAI"]].rename(columns={"LAI": "Measured LAI"})
predictions["Training-mean baseline"] = train_mean
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")

for ax, name in zip(axes, (optical_index, radar_index)):
    slope, intercept = np.polyfit(train[name], train.LAI, 1)
    predictions[name] = intercept + slope * test[name]

    x = np.linspace(table[name].min(), table[name].max(), 100)
    ax.plot(x, intercept + slope * x, label="Fitted line")
    ax.scatter(train[name], train.LAI, label="2022–2023 measurements")
    ax.scatter(test[name], test.LAI, marker="D", label="2024 measurements")
    ax.set(xlabel=name, title=f"LAI = {intercept:.2f} + ({slope:.2f}) × index")

axes[0].set_ylabel("Field-measured LAI")
axes[0].legend()
plt.show()

display(predictions.round({column: 2 for column in predictions.columns if column != "field_date"}))
errors = predictions.drop(columns=["field_date", "Measured LAI"]).sub(
    predictions["Measured LAI"], axis=0
)
display(np.sqrt(errors.pow(2).mean()).round(3).to_frame("2024 RMSE"))

# %% [markdown]
# ## 6. Compare results with another group
#
# - Which index had the stronger correlation in 2022–2023? Which had the lower prediction error in 2024? Did either beat the training-mean baseline?
# - Pick the date with the largest error for one model. How far apart were the image and the field visit? How might rapid leaf-out or a wet canopy affect the comparison?
# - All three test visits come from the same forest. What additional measurements would let you test whether an index can distinguish stands with different LAI on the same midsummer day?
#
# ## If you finish early
#
# Pick two radar indices with the same correlation. Compare their formulas in step 3. Can you write one as a rescaling of the other? If you change `tau` in RDPI, rerun the cells starting with the index definitions and see whether the tie persists.
#
# The radar formulas use linear backscatter power. The VV/VH ratio and normalized radar difference are VV/VH-based analogues of the slide deck's HH/HV-based RRVI and RRNDVI. The reduced simple ratio (RSR) uses the 2nd and 98th percentiles of usable training pixels in the SWIR1 band to scale its adjustment across all dates.
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
