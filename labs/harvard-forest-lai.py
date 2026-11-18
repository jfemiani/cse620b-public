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

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/jfemiani/cse620b-public/blob/main/labs/harvard-forest-lai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="12aee42f"
# # Can satellites tell how leafy a forest is?
#
# **Ungraded in-class lab · 25–30 minutes · Work in small groups**
#
# Compare satellite images of Harvard Forest with measurements of leaf area taken on the ground. Your group will calculate one optical index and one radar index, then test how well each predicts leaf area on three visits in 2024.
#
# Run the setup and image cells first. Write your own code in the two **Your turn** cells. If you want to check your work, use the [worked key](https://colab.research.google.com/github/jfemiani/cse620b-public/blob/main/labs/harvard-forest-lai-key.ipynb) after class.
#
# ## 1. Download the data
#
# The download is under 1 MB. The satellite rasters were cropped and aligned in advance so you can calculate indices in Colab without opening full satellite scenes or installing GIS software.
#
# | File | What's inside |
# |---|---|
# | `YYYY-MM-DD.npz` | One file per **field visit**. A compressed bundle of nine named NumPy arrays: six Sentinel-2 reflectance bands, the Sentinel-2 scene classification (`SCL`), and two Sentinel-1 radar bands. Each array is **60 rows × 60 columns** on the same 20 m grid. |
# | `footprint.npy` | One **60 × 60 Boolean array**, reused on every date. `True` means the pixel center is in the approximate field sampling area. |
# | `scenes.csv` | One row per matched visit, with the field date, both satellite dates, and the original scene IDs. |
# | `grid.json` | The coordinate reference system (UTM zone 18N), pixel-to-map transform, grid shape, and tower coordinates. |
#
# `.npy` stores one array and preserves its shape and data type; `.npz` bundles multiple named arrays. Unlike GeoTIFF, these arrays don't carry map coordinates on their own. That's why `grid.json` is separate. The field measurements come from Harvard Forest as a separate CSV, not as an image.

# %% id="0b2654f0" outputId="f7f8975d-cd0d-441f-b774-dcb3d199181e" colab={"base_uri": "https://localhost:8080/"} language="bash"
# set -e
# wget -nv -O harvard-forest-chips.zip https://vision.csi.miamioh.edu/data/cse620b/forestry/harvard-forest-chips.zip
# unzip -oq harvard-forest-chips.zip -d harvard-forest
# wget -nv -O lai.csv https://harvardforest1.fas.harvard.edu/data/p06/hf069/hf069-02-lai-site.csv

# %% [markdown] id="d67e4f76"
# ## 2. Find the study area
#
# These color views show the same 1.2 km-wide area at Harvard Forest's Prospect Hill tract in Massachusetts. North is up and west is left. The white outline marks pixels whose centers are west of the Environmental Measurement Site (EMS) tower and within 500 m of it. The half-circle approximates the area sampled by field transects; it is **not** a map of individual plots or the tower's atmospheric footprint. The outline stays fixed; clouds and invalid pixels can change which pixels you use on each date.
#
# **Look first.** Where does the forest change most between June and November? Which changes might a radar image pick up?

# %% id="84560b78" outputId="858240a1-c665-4c57-f226-44150e424579" colab={"base_uri": "https://localhost:8080/", "height": 428}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

footprint = np.load("harvard-forest/footprint.npy")

def load_bands(visit):
    with np.load(f"harvard-forest/{visit}.npz") as chip:
        # Read the named arrays before closing the archive.
        return dict(chip)

def show_rgb(ax, bands):
    rgb = np.dstack([bands["B04"], bands["B03"], bands["B02"]])
    # Brighten for display only; index calculations use the original values.
    ax.imshow(np.clip(rgb * 4, 0, 1))
    ax.contour(footprint, levels=[0.5], colors="white", linewidths=1)
    ax.set_axis_off()

fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
for ax, visit, label in zip(
    axes,
    ["2023-06-05", "2023-11-07"],
    ["June 11, 2023", "November 8, 2023"],
):
    show_rgb(ax, load_bands(visit))
    ax.set_title(label)
plt.show()

# %% [markdown] id="07f81a24"
# ## 3. Read and display the source bands
#
# Each band is a 2D array. Sentinel-2 values are surface reflectance. Sentinel-1 values are linear radar backscatter power. This radar series has **VV** and **VH**. The letters give the transmit and receive polarizations, respectively. **HH** (horizontal transmit and receive) and **HV** are other possible channels. The archive does not carry them.
#
# Run the cell to see each band and the values of one pixel. The color scale is set separately for each band. Use its colorbar when comparing values. Try changing the visit to `"2023-11-07"`. What changes in red and near-infrared? What changes in VV and VH?

# %% id="c199e5c7" outputId="92ff73ce-374a-4b6a-c4eb-55f2bac1534a" colab={"base_uri": "https://localhost:8080/", "height": 671}
visit = "2023-06-05"  # Date of the field visit used to name the image file
bands = load_bands(visit)
band_names = {
    "Red (B04)": "B04", "Green (B03)": "B03", "Blue (B02)": "B02",
    "NIR (B08)": "B08", "SWIR 1 (B11)": "B11", "SWIR 2 (B12)": "B12",
    "VV": "VV", "VH": "VH",
}

fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
for ax, (label, band) in zip(axes.flat, band_names.items()):
    image = ax.imshow(bands[band], cmap="gray")
    ax.contour(footprint, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(label)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.7)
plt.show()

row, col = 30, 20
print(f"Pixel at row {row}, column {col}")
for label, band in band_names.items():
    print(f"{label}: {bands[band][row, col]:.4f}")

# %% [markdown] id="010b5e24"
# ## 4. Match satellite images to field measurements
#
# **Leaf area index (LAI)** is leaf area per unit of ground area (m² of leaf per m² of ground). Harvard Forest's HF069 project tracks vegetation at the EMS tower. It publishes both plot-level measurements (`hf069-01`) and site-level summaries (`hf069-02`). We use the site-level table, which has one row per site and survey date.
#
# `site` identifies the sampling site; we keep `ems`. `date` is the field survey date, `n_plots` counts the sampled plots, and `all_mean` is the mean total LAI across plots. The CSV also reports spread and confidence intervals and has separate `decid_*` columns for deciduous LAI; we don't use those here.
#
# Joining to `scenes.csv` gives **12 rows**, one per matched survey: nine from 2022–2023 and three from 2024. The field and satellite dates don't have to be identical. You have **one** measured LAI value per row, not one per pixel, so summarize each index map over the usable pixels before comparing it with LAI.
#
# Find the June and November visits. Do the measured LAI values agree with your impression from the color images?

# %% id="079093a5" outputId="53562c89-8806-4066-930e-9a340a216c61" colab={"base_uri": "https://localhost:8080/", "height": 426}
field = pd.read_csv("lai.csv")
field = field.loc[field["site"] == "ems", ["date", "n_plots", "all_mean"]]
field = field.rename(columns={"date": "field_date", "all_mean": "LAI"})
scenes = pd.read_csv("harvard-forest/scenes.csv")
# Join on the field survey date, not the optical or radar acquisition date.
visits = scenes.merge(field, on="field_date", validate="one_to_one")
display(visits[["field_date", "optical_date", "radar_date", "n_plots", "LAI"]])

def usable_pixels(bands):
    # Keep vegetation (SCL class 4) inside the footprint, then use a shared
    # mask so that both indices summarize the same pixels on this visit.
    valid = footprint & (bands["SCL"] == 4)
    for band in band_names.values():
        valid &= np.isfinite(bands[band]) & (bands[band] > 0)
    return valid

# %% [markdown] id="8d950e1f"
# ## 5. Your turn · Calculate two indices
#
# 1. Pick one optical index and one radar index. For example, start with NDVI and a ratio of the two radar channels. Write down the formulas your group chose.
# 2. Implement the formulas with the source arrays. Make a map of each index for one visit. Use `usable_pixels(bands)` to keep the same pixels in both maps. Use `np.nan` for masked pixels when plotting.
# 3. Repeat for every date in `visits["field_date"]`. For each index, take the median over the usable pixels. Add the two summaries as columns in a copy of `visits`, with one row per visit.
#
# The index maps will look roughly semicircular when you mask them. The cyan outline in the source-band plots shows where that shape comes from. Gaps inside it are pixels that fail the vegetation or validity checks.

# %% id="419e17f8"
# TODO: Calculate and plot both index arrays for one visit.
# TODO: Use the Boolean mask to select the same pixels from both arrays.
# TODO: Save one median per index per field date, alongside measured LAI.

# %% [markdown] id="e6ed452e"
# ## 6. Your turn · Fit and test your models
#
# 1. On the **2022–2023** rows, make a scatterplot and calculate Pearson's r for each index versus measured LAI. Which has the stronger relationship (larger absolute r), and is it positive or negative? Use these same rows to fit a separate line for each index, with index as x and measured LAI as y. Hint: `np.polyfit` with degree 1 fits a straight line.
# 2. Use each fitted line to predict LAI from the index values for the three **2024** visits. Put the two predictions and measured LAI side by side for each date.
# 3. Compare the models' root mean squared errors (RMSE) with a baseline that predicts the 2022–2023 mean LAI for every 2024 visit. Keep the 2024 rows out of the fitting step.

# %% id="4e4a70ff"
# TODO: Split by field year before calculating correlations or fitting.
# TODO: Compare the two training correlations and fit one line per index.
# TODO: Predict the held-out LAI and compare RMSE with the training-mean baseline.

# %% [markdown] id="17e1d259"
# ## Share your results
#
# Tell another group which indices you chose and whether either model beat the baseline. Which visit was hardest to predict? Check how far apart the field and satellite dates were. How could rapid leaf-out or a wet canopy affect the result?
#
# All three test dates come from the same forest. What measurements would you need to test whether either index can distinguish two stands with different LAI on the same day?
#
# ## Data and worked examples
#
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
