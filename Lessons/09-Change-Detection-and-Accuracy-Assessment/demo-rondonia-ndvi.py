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

# %% [markdown] {"id": "rondonia-title", "language": "markdown", "slideshow": {"slide_type": "slide"}}
# # Rondonia: NDVI in 2002 Minus NDVI in 1992
#
# Reproduce the calculation in textbook Figure 15.1 using two documented
# Landsat 5 TM crops near Ariquemes, Brazil. The original caption does not
# identify its scenes or crop, so this is a methodological reproduction,
# not the exact textbook image.
#
# [Open in Colab](https://colab.research.google.com/github/jfemiani/cse620b-public/blob/main/Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-rondonia-ndvi.ipynb)

# %% [markdown] {"id": "rondonia-provenance", "language": "markdown", "slideshow": {"slide_type": "slide"}}
# ## Check the conditions before subtracting
#
# - Landsat 5 TM, Collection 2 Level-2 surface reflectance, 30 m pixels.
# - 1992-03-10: LT05_L2SP_232067_19920310_02_T2.
# - 2002-02-02: LT05_L2SP_232067_20020202_02_T1.
# - Both crops have the same grid, but Tier 2 feature registration is unverified.
# - March versus February can introduce phenology and moisture differences.
# - Cloud masking leaves only about 24% common coverage. QA_RADSAT is not
#   bundled, so this is not a complete analysis-ready or validated change map.
#
# Data: USGS Landsat via Microsoft Planetary Computer.
# [Provenance and source URLs](https://github.com/jfemiani/cse620b-public/releases/download/rondonia-ndvi-v1/provenance.json).
# In a fresh Colab runtime, run `%pip install rasterio` before the import cell.

# %% {"id": "rondonia-download", "language": "python", "slideshow": {"slide_type": "skip"}}
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)
release = "https://github.com/jfemiani/cse620b-public/releases/download/rondonia-ndvi-v1"
for year in (1992, 2002):
    filename = f"rondonia-{year}.tif"
    if not (data_dir / filename).exists():
        urlretrieve(f"{release}/{filename}", data_dir / filename)

# %% [markdown] {"id": "rondonia-formula", "language": "markdown", "slideshow": {"slide_type": "slide"}}
# ## Scale first, then compute NDVI
#
# $$\rho = 0.0000275\,\mathrm{DN}-0.2,\qquad
# \mathrm{NDVI}=\frac{\rho_{NIR}-\rho_{red}}{\rho_{NIR}+\rho_{red}}$$
#
# TM band 3 is red and band 4 is NIR. These crops store red, NIR, and QA_PIXEL
# as bands 1, 2, and 3. The additive offset does not cancel from the NDVI
# denominator. Exclude fill, dilated cloud, cloud, shadow, and snow flags;
# then retain nonnegative reflectances with a positive denominator.

# %% {"id": "rondonia-ndvi", "language": "python", "slideshow": {"slide_type": "slide"}}
ndvi = {}
transforms = []
for year in (1992, 2002):
    with rasterio.open(data_dir / f"rondonia-{year}.tif") as source:
        red_dn, nir_dn, quality = source.read()
        transforms.append(source.transform)
    red = red_dn.astype(np.float32) * 0.0000275 - 0.2
    nir = nir_dn.astype(np.float32) * 0.0000275 - 0.2
    valid = ((quality & 63) == 0) & (red_dn != 0) & (nir_dn != 0)
    valid &= (red >= 0) & (nir >= 0) & (nir + red > 0)
    values = np.full(red.shape, np.nan, dtype=np.float32)
    np.divide(nir - red, nir + red, out=values, where=valid)
    ndvi[year] = values
assert transforms[0] == transforms[1], "Pixel grids must agree before subtraction"
common = np.isfinite(ndvi[1992]) & np.isfinite(ndvi[2002])
difference = np.where(common, ndvi[2002] - ndvi[1992], np.nan)
print(f"Comparable pixels: {common.sum():,} / {common.size:,} ({common.mean():.1%})")
print(f"NDVI difference range: {np.nanmin(difference):.3f} to {np.nanmax(difference):.3f}")

# %% [markdown] {"id": "rondonia-compare", "language": "markdown", "slideshow": {"slide_type": "slide"}}
# ## Same pixels, same display scale
#
# Grey pixels are excluded on either date. The common mask is used in every
# panel so cloud boundaries do not become apparent changes in the flicker test.
# NDVI maps share [-1, 1]. Difference brightness also uses fixed limits [-1, 1]:
# zero is middle grey, positive is brighter, negative is darker. This clips
# display values outside that interval; numerical NDVI differences can reach
# [-2, 2]. The displayed limits never change between dates.

# %% {"id": "rondonia-panels", "language": "python", "slideshow": {"slide_type": "slide"}}
ndvi_cmap = plt.colormaps["RdYlGn"].with_extremes(bad="#bdbdbd")
change_cmap = plt.colormaps["gray"].with_extremes(bad="#bdbdbd")
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
for axis, year in zip(axes[:2], (1992, 2002)):
    panel = axis.imshow(np.where(common, ndvi[year], np.nan), cmap=ndvi_cmap, vmin=-1, vmax=1)
    axis.set_title(f"{year} NDVI")
fig.colorbar(panel, ax=axes[:2], shrink=0.65, label="NDVI")
panel = axes[2].imshow(difference, cmap=change_cmap, vmin=-1, vmax=1)
axes[2].set_title("2002 minus 1992")
fig.colorbar(panel, ax=axes[2], shrink=0.65, label="NDVI difference")
for axis in axes:
    axis.axis("off")
fig.savefig(output_dir / "rondonia-ndvi-comparison.png", dpi=150)
plt.show()

# %% [markdown] {"id": "rondonia-flicker", "language": "markdown", "slideshow": {"slide_type": "slide"}}
# ## Flicker between the two NDVI maps
#
# Both frames have identical dimensions, masks, and stretch. Alternate the
# exported PNGs in adjacent presentation slides, or view the animated GIF.
# Inspect persistent patterns, then ask what else besides land-cover change
# could explain them. Do not estimate cleared forest area from this sample.

# %% {"id": "rondonia-frames", "language": "python", "slideshow": {"slide_type": "slide"}}
frames = []
for year in (1992, 2002):
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(np.where(common, ndvi[year], np.nan), cmap=ndvi_cmap, vmin=-1, vmax=1)
    axis.set_title(f"Rondonia: {year} NDVI (common clear pixels)")
    axis.axis("off")
    path = output_dir / f"rondonia-{year}-ndvi.png"
    fig.savefig(path, dpi=120)
    plt.show()
    with Image.open(path) as frame:
        frames.append(frame.convert("RGB"))
frames[0].save(output_dir / "rondonia-ndvi-flicker.gif", save_all=True,
               append_images=frames[1:], duration=900, loop=0)

# %% [markdown] {"id": "rondonia-summary", "language": "markdown", "slideshow": {"slide_type": "slide"}}
# ## What this reproduces
#
# Figure 15.1's operation and sign convention: NDVI in 2002 minus NDVI in 1992,
# in Rondonia, using Landsat 5 TM. It does not reproduce the undocumented
# original footprint or acquisition dates. Better-matched cloud-free scenes,
# saturation checks, verified registration, and independent reference data are
# needed before attributing these spectral differences to deforestation.