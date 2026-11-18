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
# # Sentinel-2 Level-1C at Harvard Forest
#
# A worked look at a real Sentinel-2A multispectral image over Harvard Forest,
# Massachusetts. The scene was acquired on June 11, 2023, in tile **T18TYN**.
# We read a compact, georeferenced chip around the Harvard Forest EMS tower,
# decode its stored digital numbers (DN) with the product's radiometric
# metadata, and plot all 13 bands, including the atmospheric bands.
#
# This notebook is a demonstration, not an exercise. The prepared chip keeps
# the lesson download small; all imagery comes from the public Level-1C product.

# %% [markdown]
# ## Acquisition and provenance
#
# - **Site:** Harvard Forest EMS tower, approximately 42.5378° N, 72.1715° W.
# - **Acquisition:** Sentinel-2A MSI, 2023-06-11 15:51:19 UTC.
# - **Product:** `S2A_MSIL1C_20230611T153821_N0509_R011_T18TYN_20230611T205742.SAFE`
# - **Tile:** T18TYN, UTM zone 18N (EPSG:32618); product cloud-cover estimate
#   is about 5%.
#
# The source JP2 bands and tile metadata are available anonymously from the
# [AWS Open Data Sentinel-2 L1C bucket](https://sentinel-s2-l1c.s3.eu-central-1.amazonaws.com/tiles/18/T/YN/2023/6/11/0/).
# For example, [B04 (red)](https://sentinel-s2-l1c.s3.eu-central-1.amazonaws.com/tiles/18/T/YN/2023/6/11/0/B04.jp2)
# is a 10 m JPEG2000 image. The [product XML metadata](https://sentinel-s2-l1c.s3.eu-central-1.amazonaws.com/products/2023/6/11/S2A_MSIL1C_20230611T153821_N0509_R011_T18TYN_20230611T205742/metadata.xml)
# is the source of the quantification value and band offsets used below.
# The acquisition is also indexed as
# [`S2A_18TYN_20230611_0_L1C`](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/S2A_18TYN_20230611_0_L1C).
#
# Full 10 m JP2 files are roughly 100 MB each. To make this notebook practical
# to run in Colab, the lesson uses GeoTIFF crops around the same 2 km square.
# The original six-band chip averages the 10 m bands onto a 20 m grid.
# The additional bands are separate GeoTIFFs at their native 20 m or 60 m
# spacing. The 60 m crops extend slightly past the display boundary so their
# native pixels cover the whole square.

# %% [markdown]
# ## Level-1C, Level-2A, and reflectance
#
# Level-1C is orthorectified **top-of-atmosphere (TOA) reflectance**: it has
# not had the atmosphere removed. Level-2A applies atmospheric correction to
# estimate **bottom-of-atmosphere (BOA, or surface) reflectance** and includes
# a scene-classification layer. This sample is Level-1C, so do not interpret
# its values as surface reflectance or expect an L2A cloud mask.
#
# Sentinel-2 stores reflectance as integer DN. The product XML reports
# `QUANTIFICATION_VALUE = 10000`, `RADIO_ADD_OFFSET = -1000` for each band,
# `NODATA = 0`, and `SATURATED = 65535`. Thus, for valid pixels:
#
# \[
# \rho_{\mathrm{TOA}} =
# \frac{\mathrm{DN} + \mathrm{RADIO\_ADD\_OFFSET}}
#      {\mathrm{QUANTIFICATION\_VALUE}}
# = \frac{\mathrm{DN} - 1000}{10000}.
# \]
#
# Read those values from the actual product metadata rather than assuming
# every Sentinel-2 processing baseline uses the same offset.

# %% [markdown]
# ## Bands in this chip
#
# Wavelengths are approximate band-center wavelengths. The 10 m channels have
# been averaged over aligned 2 × 2 blocks for the shared 20 m chip grid.
#
# | Band | Spectral region | Approx. center | Native resolution | What it shows |
# | --- | --- | ---: | ---: | --- |
# | B01 | Coastal aerosol | 443 nm | 60 m | Sensitive to atmospheric scattering; useful for aerosol correction and coastal water. |
# | B02 | Blue | 490 nm | 10 m | The blue channel in natural color; sensitive to haze and useful over shallow water. |
# | B03 | Green | 560 nm | 10 m | Captures the visible green reflectance peak of many plants. |
# | B04 | Red | 665 nm | 10 m | Chlorophyll absorbs strongly here; this is the red band in NDVI. |
# | B05 | Red edge 1 | 704 nm | 20 m | Samples the start of the sharp rise from red absorption to near-infrared reflectance. |
# | B06 | Red edge 2 | 740 nm | 20 m | Samples farther up that rise; helps characterize canopy chlorophyll. |
# | B07 | Red edge 3 | 783 nm | 20 m | Samples the upper shoulder near the near-infrared plateau. |
# | B08 | Broad near-infrared | 833 nm | 10 m | Healthy vegetation often reflects strongly here; this is the NIR band in NDVI. |
# | B8A | Narrow near-infrared | 865 nm | 20 m | Samples a narrower, longer-wavelength part of the near-infrared plateau. |
# | B09 | Water vapor | 945 nm | 60 m | Samples a water-vapor absorption feature; used to estimate atmospheric water vapor. |
# | B10 | Cirrus | 1375 nm | 60 m | Water vapor suppresses much of the surface signal, so high cirrus can stand out. Level-2A omits this band. |
# | B11 | Shortwave infrared 1 | 1614 nm | 20 m | Sensitive to liquid water in plants and soil; useful for moisture mapping. |
# | B12 | Shortwave infrared 2 | 2202 nm | 20 m | Helps distinguish burned areas, exposed materials, and moisture differences. |

# %%
%pip -q install rasterio scikit-image

# %%
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import transform
from skimage.segmentation import find_boundaries
from skimage.transform import resize

bands = {
    "B01": {"name": "Coastal aerosol", "id": "0", "wavelength_nm": 443, "resolution": "60 m"},
    "B02": {"name": "Blue", "id": "1", "wavelength_nm": 490, "resolution": "10 m → 20 m mean"},
    "B03": {"name": "Green", "id": "2", "wavelength_nm": 560, "resolution": "10 m → 20 m mean"},
    "B04": {"name": "Red", "id": "3", "wavelength_nm": 665, "resolution": "10 m → 20 m mean"},
    "B05": {"name": "Red edge 1", "id": "4", "wavelength_nm": 704, "resolution": "20 m"},
    "B06": {"name": "Red edge 2", "id": "5", "wavelength_nm": 740, "resolution": "20 m"},
    "B07": {"name": "Red edge 3", "id": "6", "wavelength_nm": 783, "resolution": "20 m"},
    "B08": {"name": "Broad NIR", "id": "7", "wavelength_nm": 833, "resolution": "10 m → 20 m mean"},
    "B8A": {"name": "Narrow NIR", "id": "8", "wavelength_nm": 865, "resolution": "20 m"},
    "B09": {"name": "Water vapor", "id": "9", "wavelength_nm": 945, "resolution": "60 m"},
    "B10": {"name": "Cirrus", "id": "10", "wavelength_nm": 1375, "resolution": "60 m"},
    "B11": {"name": "SWIR 1", "id": "11", "wavelength_nm": 1614, "resolution": "20 m"},
    "B12": {"name": "SWIR 2", "id": "12", "wavelength_nm": 2202, "resolution": "20 m"},
}
band_order = list(bands)

# %% [markdown]
# ## Download the lesson chip
#
# The small archive is served outside GitHub at
# [vision.csi.miamioh.edu](https://vision.csi.miamioh.edu/data/cse620b/forestry/sentinel-2-l1c-all-bands.zip).
# This cell uses `wget` to retrieve the chip and its matching product XML.

# %%
%%bash
set -e
wget -nv -O sentinel-2-l1c-all-bands.zip \
    https://vision.csi.miamioh.edu/data/cse620b/forestry/sentinel-2-l1c-all-bands.zip
unzip -oq sentinel-2-l1c-all-bands.zip -d sentinel-2-l1c-demo
wget -nv -O sentinel-2-l2a-demo.zip https://vision.csi.miamioh.edu/data/cse620b/forestry/sentinel-2-l2a-demo.zip
unzip -oq sentinel-2-l2a-demo.zip SCL.tif -d sentinel-2-l2a-demo

# %% [markdown]
# ## Read the chip and its product metadata

# %%
product_id = "S2A_MSIL1C_20230611T153821_N0509_R011_T18TYN_20230611T205742"
chip_name = "sentinel-2-l1c-harvard-forest-2023-06-11.tif"
data_dir = Path("sentinel-2-l1c-demo") / "data"
chip_path = data_dir / chip_name
product_metadata = ElementTree.parse(data_dir / "metadata.xml").getroot()

metadata_fields = {
    element.tag.rsplit("}", 1)[-1]: element
    for element in product_metadata.iter()
}
quantification = int(metadata_fields["QUANTIFICATION_VALUE"].text)
add_offsets = {
    element.attrib["band_id"]: int(element.text)
    for element in product_metadata.iter()
    if element.tag.rsplit("}", 1)[-1] == "RADIO_ADD_OFFSET"
}
special_values = {}
for element in product_metadata.iter():
    if element.tag.rsplit("}", 1)[-1] == "Special_Values":
        values = {child.tag.rsplit("}", 1)[-1]: child.text for child in element}
        special_values[values["SPECIAL_VALUE_TEXT"]] = int(values["SPECIAL_VALUE_INDEX"])

with rasterio.open(chip_path) as chip:
    dn = {
        band: chip.read(index).astype(np.float32)
        for index, band in enumerate(("B02", "B03", "B04", "B08", "B11", "B12"), start=1)
    }
    bounds = chip.bounds
    chip_crs = chip.crs
band_bounds = dict.fromkeys(dn, bounds)
for band in bands.keys() - dn.keys():
    with rasterio.open(data_dir / f"{band}.tif") as chip:
        dn[band] = chip.read(1).astype(np.float32)
        band_bounds[band] = chip.bounds

no_data = special_values["NODATA"]
saturated = special_values["SATURATED"]
toa = {
    band: np.where(
        (dn[band] != no_data) & (dn[band] != saturated),
        (dn[band] + add_offsets[bands[band]["id"]]) / quantification,
        np.nan,
    )
    for band in band_order
}

print(f"Product: {product_id}")
print(f"{len(toa)} bands; {chip_crs}")
for band, image in toa.items():
    print(f"{band}: {image.shape[0]} rows × {image.shape[1]} columns")
print(f"Metadata: quantification={quantification}, offsets={sorted(set(add_offsets.values()))}")
print(f"Special values: {special_values}")

# %% [markdown]
# All 13 panels show the same 2 km square; the yellow cross marks the tower.
# Notice the coarser 60 m pixels in the atmospheric bands. Each panel has its
# own contrast stretch to make spatial patterns visible, even in a band with
# very little signal. Use the colorbars, not screen brightness, to compare
# reflectance across bands.

# %%
extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
tower_lon, tower_lat = -72.1715, 42.5378
tower_x, tower_y = transform("EPSG:4326", chip_crs, [tower_lon], [tower_lat])
tower_x, tower_y = tower_x[0], tower_y[0]

fig, axes = plt.subplots(4, 4, figsize=(13, 12), constrained_layout=True)
for ax in axes.flat:
    ax.set_axis_off()
for ax, band in zip(axes.flat, band_order):
    image = toa[band]
    low, high = np.nanpercentile(image, (2, 98))
    left, bottom, right, top = band_bounds[band]
    artist = ax.imshow(
        image,
        extent=(left, right, bottom, top),
        origin="upper",
        cmap="gray",
        interpolation="nearest",
        vmin=low,
        vmax=high,
    )
    ax.scatter(tower_x, tower_y, marker="+", color="yellow", edgecolor="black", s=90)
    ax.set_title(
        f'{band} · {bands[band]["name"]} · {bands[band]["wavelength_nm"]} nm\n'
        f'{bands[band]["resolution"]}'
    )
    ax.set(xlim=(bounds.left, bounds.right), ylim=(bounds.bottom, bounds.top))
    fig.colorbar(artist, ax=ax, label="TOA reflectance")
fig.suptitle("Harvard Forest — individual Sentinel-2 Level-1C bands", fontsize=15)
plt.show()

# %% [markdown]
# ## Natural-color RGB
#
# B04/B03/B02 are assigned to red/green/blue. The composite shows the visible
# landscape around the tower, marked with a yellow cross. A per-channel
# percentile stretch and gamma adjustment make it easier to see on screen;
# they are display operations, not additional reflectance corrections.

# %%
rgb = np.stack([toa[band] for band in ("B04", "B03", "B02")], axis=-1)
low, high = np.nanpercentile(rgb, (2, 98), axis=(0, 1))
rgb_display = np.clip((rgb - low) / (high - low), 0, 1) ** (1 / 2.2)

fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
ax.imshow(rgb_display, extent=extent, origin="upper")
ax.scatter(tower_x, tower_y, marker="+", color="yellow", edgecolor="black", s=110)
ax.set_title("Sentinel-2 true-color composite (B04 red, B03 green, B02 blue)")
ax.set_xlabel("UTM easting (m)")
ax.set_ylabel("UTM northing (m)")
plt.show()

# %% [markdown]
# **Interpretation:** This is a 2 km view centered on the tower, not a
# classification or a cloud-free guarantee. Level-1C values describe
# top-of-atmosphere reflectance. The NIR and SWIR channels complement visible
# color because vegetation reflects strongly in NIR while water content and
# vegetation structure influence the SWIR response.

# %% [markdown]
# ## Overlay the Level-2A scene classification
#
# Level-1C has no SCL of its own. These yellow outlines show where neighboring
# SCL pixels change class in the matching June 11 Level-2A product. The
# classification crop covers only the central 1.2 km, so this panel zooms
# to that overlap. Both layers are drawn in the same UTM coordinates.
# Nearest-neighbor resizing to 800 × 800 keeps the class codes intact.
# Finding edges on that finer display grid gives thinner outlines.

# %%
with rasterio.open("sentinel-2-l2a-demo/SCL.tif") as src:
    labels = src.read(1)
    left, bottom, right, top = src.bounds
    assert src.crs == chip_crs
display_labels = resize(labels, (800, 800), order=0, anti_aliasing=False)
edges = find_boundaries(display_labels, mode="thick")

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
ax.imshow(rgb_display, extent=extent)
ax.imshow(np.ma.masked_where(~edges, edges), extent=(left, right, bottom, top),
          cmap="Wistia", vmin=0, vmax=1, interpolation="nearest")
ax.set(xlim=(left, right), ylim=(bottom, top),
       title="Level-1C RGB with Level-2A SCL boundaries")
ax.set_axis_off()
plt.show()

# %% [markdown]
# ## Obtaining a different source scene
#
# To work with another date or location, use the
# [Copernicus Browser](https://browser.dataspace.copernicus.eu/): sign in with
# a Copernicus Data Space account, search for the area and acquisition date,
# select **Sentinel-2 → Level-1C**, and choose a scene with suitable cloud
# cover. Download the product (or the required tile and bands) as a SAFE
# product. Its `MTD_MSIL1C.xml` metadata supplies the quantification value,
# band-specific `RADIO_ADD_OFFSET` values, and special DN values. Keep the
# processing level and the matching metadata together; Level-2A uses BOA
# offsets and should not be decoded using the Level-1C formula above.
#
# For a direct public band URL, such as the B04 link above, use
# `wget -O B04.jp2 'BAND_URL'`. This downloads the entire band file.
#
# Contains modified Copernicus Sentinel data (2023).
