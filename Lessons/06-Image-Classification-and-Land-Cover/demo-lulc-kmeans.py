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
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: cse620b-shared
#     language: python
#     name: python3
# ---

# %% [markdown] {"id": "lulc-kmeans-1", "slideshow": {"slide_type": "slide"}}
# # Unsupervised LULC Mapping with K-Means
#
# Cluster satellite pixels by spectral similarity, then compare the cluster map with the labeled raster. Cluster IDs are arbitrary and are not land-cover class IDs.

# %% [markdown] {"id": "lulc-kmeans-2", "slideshow": {"slide_type": "slide"}}
# ## Data
#
# Place the GF1 image and dense label rasters beside this notebook. The course page lists their server paths and SFTP commands. This demo reads a 256 x 256 window.

# %% {"id": "lulc-kmeans-3", "slideshow": {"slide_type": "skip"}}
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

image_path = Path('GF1_PMS2_E113.7_N30.2_20160614_L1A0001642547-MSS2.tif')
label_path = Path('GF1_PMS2_E113.7_N30.2_20160614_L1A0001642547-MSS2_dense.tif')
window = Window(col_off=3500, row_off=1500, width=256, height=256)
with rasterio.open(image_path) as source:
    image = reshape_as_image(source.read((1, 2, 3), window=window))
with rasterio.open(label_path) as source:
    labels_rgb = reshape_as_image(source.read((1, 2, 3), window=window))

valid_mask = np.any(labels_rgb != 0, axis=2)
X_all = image[valid_mask].astype(np.float64)
X_train, X_test = train_test_split(X_all, test_size=0.3, random_state=42)
if len(X_train) > 10_000:
    X_train = X_train[np.random.default_rng(42).choice(len(X_train), 10_000, replace=False)]
if len(X_test) > 2_000:
    X_test = X_test[np.random.default_rng(43).choice(len(X_test), 2_000, replace=False)]
print(f'Fit pixels: {len(X_train):,}; evaluation pixels: {len(X_test):,}')

# %% [markdown] {"id": "lulc-kmeans-4", "slideshow": {"slide_type": "slide"}}
# ## Model
#
# K-Means assigns each pixel to its nearest spectral centroid. We use the number of labeled colors as K, but cluster IDs do not correspond to those colors.

# %% {"id": "lulc-kmeans-5", "slideshow": {"slide_type": "slide"}}
model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=7, random_state=42, n_init=10),
)
model.fit(X_train)
cluster_ids = model.predict(image.reshape(-1, 3).astype(np.float64)).reshape(image.shape[:2])
print(f'Cluster sizes in the fit sample: {np.bincount(model[-1].labels_)}')

# %% [markdown] {"id": "lulc-kmeans-6", "slideshow": {"slide_type": "slide"}}
# ## Inspect clusters
#
# The silhouette score measures separation in feature space; it does not measure agreement with the reference classes.

# %% {"id": "lulc-kmeans-7", "slideshow": {"slide_type": "slide"}}
from sklearn.metrics import silhouette_score

sample = X_test[:min(1_000, len(X_test))]
print(f'Silhouette score: {silhouette_score(model[0].transform(sample), model.predict(sample)):.3f}')
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(image)
axes[0].set_title('Image')
axes[1].imshow(labels_rgb)
axes[1].set_title('Reference labels')
axes[2].imshow(cluster_ids, cmap='tab10', vmin=0, vmax=9)
axes[2].set_title('K-Means cluster IDs')
for axis in axes:
    axis.axis('off')
plt.tight_layout()
plt.show()
