# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: cse620b-shared
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
# Define paths
# image_path = '/data/cse620b/ch12-data/C-megacities/Image__8bit_NirRGB/GF1_PMS2_E113.7_N30.2_20160614_L1A0001642547-MSS2.tif'
# label_path = '/data/cse620b/ch12-data/C-megacities/label_dense__color/GF1_PMS2_E113.7_N30.2_20160614_L1A0001642547-MSS2_dense.tif'
image_path =  'GF1_PMS2_E113.7_N30.2_20160614_L1A0001642547-MSS2.tif'
label_path = 'GF1_PMS2_E113.7_N30.2_20160614_L1A0001642547-MSS2_dense.tif'

# Define bounding box (row_start, row_stop), (col_start, col_stop)
row_start, row_stop = 1500, 2500
col_start, col_stop = 3500, 4500

window = ((col_start, col_stop ), (row_start, row_stop))

# Load image with bounding box
import warnings; warnings.filterwarnings('ignore', 'Dataset has no geotransform')
with rasterio.open(image_path) as src:
    image = src.read((1,2,3), window=window)
    image = reshape_as_image(image)
    print(f'Image shape: {image.shape}')  # (height, width, bands)

# Load labels with bounding box
with rasterio.open(label_path) as src:
    labels_rgb = src.read(window=window)
    labels_rgb = reshape_as_image(labels_rgb)
    print(f'Labels shape: {labels_rgb.shape}')  # (height, width, channels)

# %%
# Display the image
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(image)
plt.title('Sample Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(labels_rgb)
plt.title('Sample Labels')
plt.axis('off')

# %%
# Define color to class mapping based on the dataset
color_to_name = {
    (0, 0, 0): 'unlabeled',
    (200, 0, 0): 'industrial_area',
    (0, 200, 0): 'paddy_field',
    (150, 250, 0): 'irrigated_field',
    (150, 200, 150): 'dry_cropland',
    (200, 0, 200): 'garden_land',
    (150, 0, 250): 'arbor_forest',
    (150, 150, 250): 'shrub_forest',
    (200, 150, 200): 'park',
    (250, 200, 0): 'natural_meadow',
    (200, 200, 0): 'artificial_meadow',
    (0, 0, 200): 'river',
    (250, 0, 150): 'urban_residential',
    (0, 150, 200): 'lake',
    (0, 200, 250): 'pond',
    (150, 200, 250): 'fish_pond',
    (250, 250, 250): 'snow',
    (200, 200, 200): 'bareland',
    (200, 150, 150): 'rural_residential',
    (250, 200, 150): 'stadium',
    (150, 150, 0): 'square',
    (250, 150, 150): 'road',
    (250, 150, 0): 'overpass',
    (250, 200, 250): 'railway_station',
    (200, 150, 0): 'airport',
}

label_colors = np.array(list(color_to_name.keys()))
label_names = np.array(list(color_to_name.values()))
color_to_label = {tuple(color): idx for idx, color in enumerate(label_colors)}

labels =  np.apply_along_axis(lambda x: color_to_label.get(tuple(x), 0), axis=2, arr=labels_rgb)

UNLABELED = 0

# %%
mask = labels != UNLABELED

X = image[mask, :]
Y = labels[mask]

print(f'X shape: {X.shape}')
print("Y shape:", Y.shape)

plt.imshow(mask)
plt.title('Masked Labels')
plt.axis('off')
plt.show()


# %%
# Create used_labels (the color codes) and used_names (the class names)
used_labels = np.unique(Y)
used_names = label_names[used_labels]

print(f'Used Labels (colors): {used_labels}')
print(f'Used Names (class names): {used_names}')

# %%
# Split into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
print(f'Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}')

# %%
# Evaluate

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


def evaluate_classification(model, name, test_preds=None):
    # Use globals X_test, y_test, image, mask

    if test_preds is None:
        test_preds = model.predict(X_test)

    print(f"{name} Classification Report:")
    print(classification_report(y_test, test_preds, 
                                labels=used_labels,
                                target_names=label_names[used_labels],
                                zero_division=np.nan)) 
    print(f'Accuracy: {accuracy_score(y_test, test_preds):.4f}')

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred_knn)

    # Display confusion matrix using Seaborn's heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", norm='log',
                xticklabels=label_names[used_labels], 
                yticklabels=label_names[used_labels],
                annot_kws=dict(size=6))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Display the image...
    print("Predicting the labels for the entire image...")
    pred_labels = model.predict(image.reshape(-1, image.shape[2])).reshape(image.shape[:2])
    pred_rgb = label_colors[pred_labels]
    pred_rgb[~mask] = 0 # Set unlabeled pixels to black

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow(labels_rgb)
    plt.title('True Labels')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(pred_rgb)
    plt.title('Predicted Labels')
    plt.axis('off')

    
evaluate_classification(knn_model, 'KNN', test_preds=y_pred_knn) # Takes a minute to run
