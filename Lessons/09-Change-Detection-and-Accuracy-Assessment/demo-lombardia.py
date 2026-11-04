# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"id":"lombardia-title","language":"markdown","slideshow":{"slide_type":"slide"}}
# # Lombardia: One Patch, Twelve Dates
#
# Follow Sentinel-2 patch 1208 in Lombardia3 through 2019: RGB images,
# differences, spectral distances, paired-pixel correlation, and a trajectory.
# [Open in Colab](https://colab.research.google.com/github/jfemiani/cse620b-public/blob/main/Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.ipynb).
# Install rasterio in a fresh Colab runtime with `%pip install rasterio`.
#
# Source: Gallo et al., Sentinel-2 Image Time Series for Crop Mapping,
# https://www.kaggle.com/datasets/ignazio/sentinel2-crop-mapping (CC0).
# The author's RGB exporter uses stored channels 3, 2, 1 and scale 0.0001:
# https://github.com/mattiagatti/sits-crop-segmentation-models/blob/main/misc/export_samples_rgb.py
# We use the confirmed red channel, not an assumed NIR channel. Nodata is
# masked; these nine-band chips provide no separate cloud-quality band.
# These are observations of agricultural appearance, without dated disturbance labels.

# %% {"id":"lombardia-load","language":"python","slideshow":{"slide_type":"skip"}}
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

dates = ["20190308", "20190313", "20190323", "20190412", "20190512", "20190611",
         "20190711", "20190810", "20190909", "20191009", "20191108", "20191208"]
data_dir = Path("data/lombardia")
data_dir.mkdir(parents=True, exist_ok=True)
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)
base = "https://www.kaggle.com/api/v1/datasets/download/ignazio/sentinel2-crop-mapping/"
images = []
grids = []
for date in dates:
    filename = data_dir / f"{date}.tif"
    if not filename.exists():
        urlretrieve(base + f"lombardia3%2Fdata2019%2F1208%2F{date}.tif", filename)
    with rasterio.open(filename) as source:
        values = source.read(masked=True).astype(float).filled(np.nan) * 0.0001
        grids.append((source.crs, source.transform, source.shape))
    images.append(values)
assert all(grid == grids[0] for grid in grids), "Dates must share a pixel grid"
images = np.stack(images)
rgb = np.clip(images[:, [2, 1, 0]].transpose(0, 2, 3, 1) / 0.3, 0, 1) ** 0.5

# %% [markdown] {"id":"lombardia-sequence-text","language":"markdown","slideshow":{"slide_type":"slide"}}
# ## Follow the same fields through clear observations
#
# Images 1, 2, 3, 6, 8, and 9 from the twelve-date list are the visually clear
# observations. Every panel uses the same RGB stretch. Each chip covers 480 by
# 480 metres on a 10 m grid; some Sentinel-2 channels have coarser native resolution.

# %% {"id":"lombardia-sequence","language":"python","slideshow":{"slide_type":"slide"}}
sequence_indices = [0, 1, 2, 5, 7, 8]
sequence_dates = [dates[index] for index in sequence_indices]
sequence_rgb = rgb[sequence_indices]
fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
for axis, date, frame in zip(axes.flat, sequence_dates, sequence_rgb):
    axis.imshow(frame)
    axis.set_title(datetime.strptime(date, "%Y%m%d").strftime("%b %d, %Y"))
    axis.axis("off")
fig.savefig(output_dir / "lombardia-sequence.png", dpi=150)
plt.show()
frames = [Image.fromarray(np.uint8(frame * 255)).resize((480, 480)) for frame in sequence_rgb]
frames[0].save(output_dir / "lombardia-flicker.gif", save_all=True,
               append_images=frames[1:], duration=700, loop=0)

# %% [markdown] {"id":"lombardia-difference-text","language":"markdown","slideshow":{"slide_type":"slide"}}
# ## Two dates: signed differences and multiband distances
#
# March 13 minus March 8. The red difference uses symmetric limits around zero;
# blue means decrease and red means increase. The two RGB views share a stretch.
# Magnitude combines all nine stored channels. Spectral angle compares their
# proportions. These summarize measured differences; interpret causes using imagery.

# %% {"id":"lombardia-difference","language":"python","slideshow":{"slide_type":"slide"}}
before, after = images[:2]
common = np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
red_difference = np.where(common, after[2] - before[2], np.nan)
magnitude = np.where(common, np.linalg.norm(after - before, axis=0), np.nan)
norm_product = np.linalg.norm(before, axis=0) * np.linalg.norm(after, axis=0)
cosine = np.divide((before * after).sum(axis=0), norm_product,
                   out=np.full(common.shape, np.nan), where=common & (norm_product > 0))
angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
limit = np.nanmax(np.abs(red_difference))
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
axes[0].imshow(rgb[0])
axes[0].set_title("March 8, 2019")
axes[1].imshow(rgb[1])
axes[1].set_title("March 13, 2019")
panel = axes[2].imshow(red_difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
axes[2].set_title("Red: March 13 minus March 8")
fig.colorbar(panel, ax=axes[2], shrink=0.7, label="Scaled red-band difference")
for axis in axes:
    axis.axis("off")
fig.savefig(output_dir / "lombardia-difference.png", dpi=150)
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for axis, values, title in zip(axes, (magnitude, angle),
                               ("Nine-channel magnitude", "Spectral angle (degrees)")):
    panel = axis.imshow(values, cmap="viridis", vmin=0)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(panel, ax=axis, shrink=0.75)
fig.savefig(output_dir / "lombardia-cva.png", dpi=150)
plt.show()

# %% [markdown] {"id":"lombardia-correlation-text","language":"markdown","slideshow":{"slide_type":"slide"}}
# ## Paired pixels, trend line, and cross-correlation
#
# Each point is one valid location's red value on the two dates. Pearson r,
# slope, and intercept summarize this patch as one analysis region. The dashed
# identity line marks equal values. This is the book's regional calculation,
# repeated on the same patch used for the images and differences above.

# %% {"id":"lombardia-correlation","language":"python","slideshow":{"slide_type":"slide"}}
first = before[2][common]
second = after[2][common]
correlation = np.corrcoef(first, second)[0, 1]
slope, intercept = np.polyfit(first, second, 1)
line_values = np.linspace(min(first.min(), second.min()), max(first.max(), second.max()), 100)
fig, axis = plt.subplots(figsize=(7, 5), layout="constrained")
axis.scatter(first, second, s=8, alpha=0.3, edgecolors="none", color="#216b86")
axis.plot(line_values, line_values, "--", color="#666666", label="Equal values")
axis.plot(line_values, slope * line_values + intercept, color="#b2182b",
          label=f"y = {slope:.3f}x {intercept:+.3f}")
axis.set(xlabel="March 8 red", ylabel="March 13 red",
         title=f"Lombardia: r = {correlation:.3f}, n = {first.size:,}")
axis.legend()
axis.grid(alpha=0.2)
fig.savefig(output_dir / "lombardia-correlation.png", dpi=150)
print(f"r={correlation:.6f}; slope={slope:.6f}; intercept={intercept:.6f}")
plt.show()

# %% [markdown] {"id":"lombardia-trajectory-text","language":"markdown","slideshow":{"slide_type":"slide"}}
# ## From an image sequence to a trajectory
#
# Track the median red value of the marked 12 by 12 pixel window across actual
# acquisition dates. The same window is used every time. The curve summarizes
# appearance through the year; RGB images help interpret vegetation, soil,
# and atmospheric contributions. For this teaching crop, visual RGB inspection
# excludes April 12, May 12, July 11, October 9, November 8, and December 8
# because clouds/haze obscure the window. This manual selection is explicit;
# a production analysis should use a documented quality mask. Gaps remain visible.

# %% {"id":"lombardia-trajectory","language":"python","slideshow":{"slide_type":"slide"}}
times = [datetime.strptime(date, "%Y%m%d") for date in dates]
trajectory = np.nanmedian(images[:, 2, 18:30, 18:30], axis=(1, 2))
excluded_dates = {"20190412", "20190512", "20190711", "20191009", "20191108", "20191208"}
trajectory[np.array([date in excluded_dates for date in dates])] = np.nan
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained",
                         gridspec_kw={"width_ratios": [1, 2]})
axes[0].imshow(rgb[0])
axes[0].add_patch(plt.Rectangle((17.5, 17.5), 12, 12, fill=False, edgecolor="yellow", linewidth=2))
axes[0].set_title("Fixed sampling window")
axes[0].axis("off")
axes[1].plot(times, trajectory, "o-", color="#216b86")
axes[1].set(ylabel="Median scaled red value", xlabel="Acquisition date (2019)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].grid(alpha=0.2)
fig.savefig(output_dir / "lombardia-trajectory.png", dpi=150)
plt.show()