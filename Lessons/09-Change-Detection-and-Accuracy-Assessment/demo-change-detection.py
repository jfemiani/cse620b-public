# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"id": "change-title", "slideshow": {"slide_type": "slide"}}
# # Bitemporal Change Detection
#
# [Open this notebook in Colab](https://colab.research.google.com/github/jfemiani/cse620b-public/blob/main/Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-change-detection.ipynb)
#
# Compare two same-season Landsat scenes over Knoxville. The small, four-band
# tiles are derived from the OpenGeoAI GeoAI tutorial data; they are not labels
# or ground truth. This demo shows how differencing and change-vector magnitude
# answer related but different questions.

# %% [markdown] {"id": "change-data", "slideshow": {"slide_type": "slide"}}
# ## Download the image pair
#
# The files are cropped to a shared 512 x 512 window. Band order is blue,
# green, red, near infrared. Original Landsat scenes are from Source Cooperative.

# %% {"id": "change-install", "slideshow": {"slide_type": "skip"}}
%pip -q install rasterio

# %% {"id": "change-download", "slideshow": {"slide_type": "skip"}}
!wget -q -O knoxville-2023.tif https://github.com/jfemiani/cse620b-public/releases/download/week10-data-v1/knoxville-2023.tif
!wget -q -O knoxville-2024.tif https://github.com/jfemiani/cse620b-public/releases/download/week10-data-v1/knoxville-2024.tif

# %% [markdown] {"id": "change-read", "slideshow": {"slide_type": "slide"}}
# ## Read the aligned scenes
#
# The source tutorial uses Landsat band 5 (NIR) and band 4 (red). In these
# cropped files, those are bands 4 and 3 because the selected bands are stored
# in blue, green, red, NIR order.

# %% {"id": "change-load", "slideshow": {"slide_type": "skip"}}
import matplotlib.pyplot as plt
import numpy as np
import rasterio

with rasterio.open("knoxville-2023.tif") as source:
    before = source.read()
    valid = source.dataset_mask() > 0
    before_transform = source.transform
    crs = source.crs

with rasterio.open("knoxville-2024.tif") as source:
    after = source.read()
    valid &= source.dataset_mask() > 0
    assert source.transform == before_transform

print(f"Shape: {before.shape}; valid pixels: {valid.sum():,}")
print(f"CRS: {crs}")

# %% [markdown] {"id": "change-visual", "slideshow": {"slide_type": "slide"}}
# ## Compare the dates
#
# NIR-red-green false color makes vegetation conspicuous. Use one shared
# display stretch so brightness changes are not hidden by per-date scaling.

# %% {"id": "change-plot-dates", "slideshow": {"slide_type": "slide"}}
false_color_before = np.stack((before[3], before[2], before[1]), axis=-1)
false_color_after = np.stack((after[3], after[2], after[1]), axis=-1)
low, high = np.percentile(
    np.concatenate((false_color_before, false_color_after), axis=0),
    [2, 98],
    axis=(0, 1),
)

def false_color(image):
    return np.clip((image - low) / (high - low), 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(false_color(false_color_before))
axes[0].set_title("2023: NIR / red / green")
axes[1].imshow(false_color(false_color_after))
axes[1].set_title("2024: NIR / red / green")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown] {"id": "change-difference", "slideshow": {"slide_type": "slide"}}
# ## Difference one band
#
# A difference is signed: positive values mean the later image is brighter in
# that band. This robust threshold is only a classroom heuristic, not a
# validated change product. Clouds, shadows, registration error, and
# phenology can all look like change.

# %% {"id": "change-diff-code", "slideshow": {"slide_type": "slide"}}
nir_difference = after[3] - before[3]
center = np.median(nir_difference[valid])
mad = np.median(np.abs(nir_difference[valid] - center))
threshold = 3 * 1.4826 * mad
change_mask = np.abs(nir_difference - center) > threshold

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
image = axes[0].imshow(nir_difference, cmap="RdBu", vmin=-threshold * 3, vmax=threshold * 3)
axes[0].set_title("NIR difference (2024 - 2023)")
fig.colorbar(image, ax=axes[0], shrink=0.75)
axes[1].imshow(change_mask & valid, cmap="Reds")
axes[1].set_title(f"Robust outlier mask: {(change_mask & valid).sum() / valid.sum():.1%} of valid pixels")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()
print(f"Median difference: {center:.4f}; robust threshold: {threshold:.4f}")

# %% [markdown] {"id": "change-cva", "slideshow": {"slide_type": "slide"}}
# ## Add a second band: change-vector magnitude
#
# CVA magnitude summarizes the size of the change vector in red-NIR space.
# It does not tell us what caused the change; the signed band differences help
# interpret its direction.

# %% {"id": "change-cva-code", "slideshow": {"slide_type": "slide"}}
red_change = after[2] - before[2]
nir_change = after[3] - before[3]
cva_magnitude = np.hypot(red_change, nir_change)
cva_threshold = np.percentile(cva_magnitude[valid], 95)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
image = axes[0].imshow(cva_magnitude, cmap="magma")
axes[0].set_title("Red-NIR CVA magnitude")
fig.colorbar(image, ax=axes[0], shrink=0.75)
axes[1].imshow((cva_magnitude > cva_threshold) & valid, cmap="Reds")
axes[1].set_title("Top 5% of valid magnitudes")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown] {"id": "change-discuss", "slideshow": {"slide_type": "slide"}}
# ## Interpret before naming change
#
# 1. Which locations are flagged by both methods? Which are not?
# 2. What non-surface causes could explain the bright differences?
# 3. What additional data or preprocessing would you need before reporting
#    land-cover change?
#
# Source: OpenGeoAI's public GeoAI tutorial Landsat pair, cropped for this demo.
# The underlying Landsat observations are from the USGS Landsat program.