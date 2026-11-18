# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sentinel-1: see all four radar polarizations
#
# **Worked demo.** Sentinel-1A imaged Prospect Hill, Massachusetts, on June 5, 2023, the same day as the field visit in the Harvard Forest LAI lab. Radar sends out microwave pulses and measures the returning signal. These images show backscatter, not color or sunlight reflected from leaves.
#
# A Level-1 Ground Range Detected (GRD) product contains detected radar magnitude; it does not retain the complex phase. The radiometrically terrain-corrected **RTC** product here is derived from GRD. It is calibrated and terrain corrected, including a correction for the effect of slope on the illuminated area. Missing data, radar shadows, and layover are not recovered by that correction.
#
# Sentinel-1 uses one microwave wavelength band (**C-band**). VV, VH, HH, and HV are polarization channels, not different wavelengths. V and H indicate the orientation of the electric field; the first letter is what the antenna transmits, and the second is what it receives. Matching letters are **co-polarized**; different letters are **cross-polarized**.
#
# | Raster | Transmit → receive | What it measures |
# |---|---|---|
# | VV | Vertical → vertical | Co-polarized return. Roughness, moisture, and viewing geometry all affect it. |
# | VH | Vertical → horizontal | Cross-polarized return. Branches and other canopy structures can change the polarization; VH is often weaker than VV. |
# | HH | Horizontal → horizontal | Co-polarized return. Useful for mapping ice and water, although wind, roughness, and viewing angle affect brightness. |
# | HV | Horizontal → vertical | Cross-polarized return. Irregular structures, including deformed ice and vegetation, can redirect some energy into this channel. |
#
# Sentinel-1 records a single co-polarized channel or a pair with the same transmit polarization (VV+VH or HH+HV). It does not collect all four together. Harvard Forest supplies **VV+VH**; a second acquisition over western Greenland on June 2, 2023 supplies **HH+HV**. Compare the channels **within each scene**: differences between the two sites cannot be attributed to polarization alone.

# %%
%pip -q install rasterio scikit-image

# %% [markdown]
# ## Download both samples
#
# Each Vision ZIP holds two cropped, georeferenced GeoTIFFs and a `provenance.json` file identifying the source scene and assets.

# %%
%%bash
set -e
wget -nv -O sentinel-1-rtc-demo.zip https://vision.csi.miamioh.edu/data/cse620b/forestry/sentinel-1-rtc-demo.zip
unzip -oq sentinel-1-rtc-demo.zip -d sentinel-1-rtc-demo
wget -nv -O sentinel-1-rtc-hh-hv.zip https://vision.csi.miamioh.edu/data/cse620b/forestry/sentinel-1-rtc-hh-hv.zip
unzip -oq sentinel-1-rtc-hh-hv.zip -d sentinel-1-rtc-hh-hv
wget -nv -O sentinel-2-l2a-demo.zip https://vision.csi.miamioh.edu/data/cse620b/forestry/sentinel-2-l2a-demo.zip
unzip -oq sentinel-2-l2a-demo.zip SCL.tif -d sentinel-2-l2a-demo

# %%
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.segmentation import find_boundaries

# %% [markdown]
# ## Read a small window from each raster
#
# Rasterio reads band 1 of each cropped GeoTIFF. Within each scene, the two rasters cover the same area on the same grid. `masked=True` preserves the raster's nodata mask. The arrays contain **linear backscatter power**. We use those linear values for the Harvard Forest VV/VH ratio and convert positive values to decibels for the display.

# %%
power = {}
grids = {}
for folder, polarizations in [
    ("sentinel-1-rtc-demo", ("vv", "vh")),
    ("sentinel-1-rtc-hh-hv", ("hh", "hv")),
]:
    for polarization in polarizations:
        with rasterio.open(f"{folder}/{polarization}.tif") as src:
            values = src.read(1, masked=True)
            power[polarization] = np.ma.masked_less_equal(np.ma.masked_invalid(values), 0)
            grids[polarization] = (src.crs, src.bounds)
            print(f"{polarization.upper()}: {src.crs}, {src.res} m, crop {values.shape}")

# %% [markdown]
# ## Compare the channels on the same display scale
#
# Plotting the two channels from a scene on the same dB scale lets you see which return is stronger. The conversion is `10 × log10(power)`: for example, a linear power of 0.01 is −20 dB. The granular texture is partly **speckle**, caused by interference among returns within a pixel. In the Harvard Forest ratio panel, values above 1 mean VV is stronger than VH. The ratio is not a direct measurement of leaf area.

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
for ax, polarization, site in zip(
    axes.flat, ("vv", "vh", "hh", "hv"),
    ("Harvard Forest", "Harvard Forest", "Greenland", "Greenland"),
):
    image = ax.imshow(10 * np.ma.log10(power[polarization]), cmap="gray", vmin=-28, vmax=0)
    ax.set_title(f"{site} | {polarization.upper()}")
    ax.set_axis_off()
fig.colorbar(image, ax=axes, shrink=0.65, label="Backscatter (dB)")
plt.show()

ratio = power["vv"] / power["vh"]
fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
image = ax.imshow(ratio, cmap="viridis", vmin=0, vmax=15)
ax.set_title("Harvard Forest | VV / VH (linear power)")
ax.set_axis_off()
fig.colorbar(image, ax=ax, shrink=0.65, label="Power ratio")
plt.show()

# %% [markdown]
# ## Overlay optical scene-class boundaries on the Harvard Forest radar
#
# Sentinel-1 does not produce Sentinel-2 SCL. The yellow edges come from the
# June 11 optical image, six days after the June 5 radar acquisition. This
# overlay gives spatial context, not same-day radar labels. The Greenland
# scene has no matching SCL in this demo, so it remains unannotated.

# %%
with rasterio.open("sentinel-2-l2a-demo/SCL.tif") as src:
    assert (src.crs, src.bounds) == grids["vv"] == grids["vh"]
    edges = find_boundaries(src.read(1), mode="thick")
    left, bottom, right, top = src.bounds

extent = (left, right, bottom, top)
fig, axes = plt.subplots(1, 2, figsize=(7, 4), layout="constrained")
for ax, polarization in zip(axes, ("vv", "vh")):
    ax.imshow(10 * np.ma.log10(power[polarization]), extent=extent,
              cmap="gray", vmin=-28, vmax=0)
    ax.imshow(np.ma.masked_where(~edges, edges), extent=extent,
              cmap="Wistia", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{polarization.upper()} with optical SCL boundaries")
    ax.set_axis_off()
plt.show()

# %% [markdown]
# ## Where to get more imagery
#
# In the [ASF Vertex search portal](https://search.asf.alaska.edu/), draw an area of interest, set the dates, and filter by the polarization pair you need. Check the asset list before downloading: a scene with VV and VH has no HH or HV to recover. For terrain-corrected imagery, use the [Planetary Computer RTC collection](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) used here, or look for an RTC product in Vertex. Verify the units, grid spacing, and orbit in the product metadata. Once you have a direct or signed URL, use `wget -O vv.tif 'ASSET_URL'`; refresh expired links and don't share signed URLs.
#
# Source: [Sentinel-1 RTC on Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) and the [Copernicus Sentinel-1 processing guide](https://sentiwiki.copernicus.eu/web/s1-processing).
# Contains modified Copernicus Sentinel data (2023).
