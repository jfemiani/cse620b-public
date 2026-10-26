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

# %% [markdown] {"id": "lulc-knn-1", "slideshow": {"slide_type": "slide"}}
# # LULC Classification with K-Nearest Neighbors
#
# Compare each labeled satellite pixel with nearby training pixels. This standalone demo trains KNN, reports held-out performance, and maps a small image crop.

# %% [markdown] {"id": "lulc-knn-2", "slideshow": {"slide_type": "slide"}}
# ## Data
#
# Place the GF1 image and dense label rasters beside this notebook. The course page lists the server paths and SFTP commands. This demo reads a 256 x 256 window and samples at most 10,000 labeled pixels for training.

# %% {"id": "lulc-knn-3", "slideshow": {"slide_type": "skip"}}
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window
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

label_colors = np.array([
    [0, 0, 0], [200, 0, 0], [0, 200, 0], [150, 250, 0], [150, 200, 150],
    [200, 0, 200], [150, 0, 250], [150, 150, 250], [200, 150, 200],
    [250, 200, 0], [200, 200, 0], [0, 0, 200], [250, 0, 150], [0, 150, 200],
    [0, 200, 250], [150, 200, 250], [250, 250, 250], [200, 200, 200],
    [200, 150, 150], [250, 200, 150], [150, 150, 0], [250, 150, 150],
    [250, 150, 0], [250, 200, 250], [200, 150, 0],
], dtype=np.uint8)
label_names = np.array('unlabeled industrial_area paddy_field irrigated_field dry_cropland garden_land arbor_forest shrub_forest park natural_meadow artificial_meadow river urban_residential lake pond fish_pond snow bareland rural_residential stadium square road overpass railway_station airport'.split())

labels = np.zeros(labels_rgb.shape[:2], dtype=np.uint8)
for class_id, color in enumerate(label_colors):
    labels[np.all(labels_rgb == color, axis=2)] = class_id

valid_mask = labels != 0
X_all = image[valid_mask].astype(np.float32)
y_all = labels[valid_mask]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)
if len(X_train) > 10_000:
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=10_000, random_state=42, stratify=y_train
    )
if len(X_test) > 2_000:
    X_test, _, y_test, _ = train_test_split(
        X_test, y_test, train_size=2_000, random_state=42, stratify=y_test
    )
used_labels = np.unique(y_all)
print(f'Training pixels: {len(y_train):,}; test pixels: {len(y_test):,}')

# %% [markdown] {"id": "lulc-knn-4", "slideshow": {"slide_type": "slide"}}
# ## Model
#
# KNN assigns the majority class among the nearest training pixels. Standardizing bands prevents a channel's numeric scale from dominating Euclidean distance.

# %% {"id": "lulc-knn-5", "slideshow": {"slide_type": "slide"}}
from sklearn.neighbors import KNeighborsClassifier

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights='distance'),
)
model.fit(X_train, y_train)

# %% [markdown] {"id": "lulc-knn-6", "slideshow": {"slide_type": "slide"}}
# ## Evaluate and inspect a prediction
#
# The report uses held-out labeled pixels. The final panel predicts a 64 x 64 crop to keep this demonstration fast.

# %% {"id": "lulc-knn-7", "slideshow": {"slide_type": "slide"}}
from sklearn.metrics import accuracy_score, classification_report

predictions = model.predict(X_test)
print(f'Pixel accuracy: {accuracy_score(y_test, predictions):.3f}')
print(classification_report(
    y_test,
    predictions,
    labels=used_labels,
    target_names=label_names[used_labels],
    zero_division=0,
))

crop = image[:64, :64]
crop_labels = labels[:64, :64]
crop_predictions = model.predict(crop.reshape(-1, 3)).reshape(crop_labels.shape)
predicted_rgb = label_colors[crop_predictions].copy()
predicted_rgb[crop_labels == 0] = 0

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(crop)
axes[0].set_title('Image')
axes[1].imshow(labels_rgb[:64, :64])
axes[1].set_title('Reference labels')
axes[2].imshow(predicted_rgb)
axes[2].set_title('KNN prediction')
for axis in axes:
    axis.axis('off')
plt.tight_layout()
plt.show()
