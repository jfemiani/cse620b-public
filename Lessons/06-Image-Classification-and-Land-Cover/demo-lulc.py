# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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
plt.imshow(labels_rgb, alpha=0.5)
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

labels =  np.apply_along_axis(lambda x: color_to_label.get((x[0], x[1], x[2]), 0), axis=2, arr=labels_rgb)

UNLABELED = np.where(label_names == 'unlabeled')[0].item()

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

# %% [markdown]
# ## Land-Use and Land-Cover Classification
#
# This notebook compares unsupervised clustering with supervised classifiers
# using labeled satellite pixels. The saved outputs below are from a prior run;
# the input rasters are not included in this repository.
#
# It is important to look at the distribution of the data to understand its characteristics and identify any patterns or anomalies.

# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

feature_names = ['Red', 'Green', 'Blue']
subset = np.random.choice(len(X_train), size=5000, replace=False)

# Convert X_train and y_train into a DataFrame for Seaborn
df = pd.DataFrame(X_train[subset], columns=feature_names)
df['Label'] = label_names[y_train[subset]]

palette = {name:color for name, color in zip(label_names, label_colors/255.0)}

# Create the pairplot using Seaborn
warnings.filterwarnings('ignore', 'The palette list has more values')  # Suppress warning about unused labels
g = sns.pairplot(df, hue='Label', palette=palette, diag_kind='hist', 
                 plot_kws={'s': 10, 'alpha': 1})  # Customize point size and transparency


# Show the plot
plt.show()


# %% [markdown]
# ### Unsupervised Classification: K-Means Clustering
#
# - K-Means partitions data into K clusters based on feature similarity
# - Uses Euclidean distance by default
# - Best when data forms equally sized round clusters....

# %%
from sklearn.cluster import KMeans

# %%
# Define number of clusters (e.g., number of classes)
k = 7 

# %%
from sklearn.pipeline import make_pipeline

# Initialize K-Means
scaler = StandardScaler()
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_model = make_pipeline(scaler, kmeans)


# %%
# Fit K-Means on training data
kmeans_model.fit(X_train)

# %%
# Predict clusters on test data
clusters = kmeans_model.predict(X_test)

# %%
preds = kmeans_model.predict(image.reshape(-1, image.shape[2]))
plt.imshow(preds.reshape(image.shape[:2]), cmap='jet')

# %%
from sklearn.metrics import silhouette_score
test_subset = np.random.choice(len(X_test), size=1000, replace=False)
print("Silhouette Score:",silhouette_score(scaler.transform(X_test[test_subset]), clusters[test_subset]))

# %% [markdown]
# ### Supervised Classification: K-Nearest Neighbors (KNN)
#
# - KNN classifies a sample based on majority vote of its neighbors
# - Distance measures: Euclidean (L2), Manhattan (L1), etc.

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
# Initialize KNN with Euclidean distance
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model = make_pipeline(scaler, knn, verbose=True)

# %%
# Fit KNN
knn_model.fit(X_train, y_train)

# %%
# Predict on test data
y_pred_knn = knn_model.predict(X_test)  #Takes about 10s to run

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

# %%
probs = knn_model.predict_proba(image.reshape(-1, image.shape[2]))
probs.shape

# %%
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=used_labels, y=y_train)
class_weights

# %%
scaled_probs = probs * class_weights
scaled_label_indices = np.argmax(scaled_probs, axis=1)
scaled_labels = used_labels[scaled_label_indices].reshape(image.shape[:2])

# %%
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(labels_rgb)
plt.title('True Labels')
plt.axis('off')
plt.subplot(122)
plt.imshow(label_colors[scaled_labels]*mask[...,None])
plt.title('Predicted Labels')
plt.axis('off')


# %%
# # %conda install -y -c conda-forge pydensecrf

# %%
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# %%
d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], len(used_labels))

unary = unary_from_softmax(scaled_probs.T)  # (7, 1000000)

# ISing model
d.setUnaryEnergy(unary.copy())
d.addPairwiseGaussian(sxy=5, compat=15)
d.addPairwiseBilateral(sxy=2, srgb=15, rgbim=image.copy(), compat=25)

Q = d.inference(1) # num iterations

crf_labels = used_labels[np.argmax(Q, axis=0)].reshape(image.shape[:2])


# %%
plt.figure(figsize=(15,4))
plt.subplot(131)
plt.imshow(labels_rgb)
plt.title('True Labels')
plt.axis('off')
plt.subplot(132)
plt.imshow(label_colors[scaled_labels]*mask[...,None])
plt.title('Per-Pixel Labels')
plt.axis('off')
plt.subplot(133)
plt.imshow(label_colors[crf_labels]*mask[...,None])
plt.title('CRF Labels')
plt.axis('off')
plt.show()

# %% [markdown]
# ### Supervised Classification: Naive Bayes Classifier
#
# **Overview:**
# - Assumes feature independence
# - Suitable for high-dimensional data

# %%
from sklearn.naive_bayes import GaussianNB

# Initialize Gaussian Naive Bayes
nb = GaussianNB()

nb_model = make_pipeline(scaler, nb, verbose=True)
nb_model.steps

# %%

to_used_index = np.zeros(len(label_names), dtype=int)
to_used_index[used_labels] = np.arange(len(used_labels))
sample_weights = class_weights[to_used_index[y_train]]


# Fit Naive Bayes
nb_model.fit(X_train, y_train, gaussiannb__sample_weight=sample_weights)

# %%
# Visualize the Gaussian distribution for each class

from scipy.stats import norm

# Select a single feature (e.g., the first feature, index 0)
feature_index = 0

X_trian_normalized = scaler.transform(X_train)[:, feature_index]

# Generate a range of values along this axis for visualization
x_axis = np.linspace(min(X_trian_normalized), max(X_trian_normalized), 100)


plt.figure(figsize=(10,4))

# Plot the Gaussian distribution for each class
for i, class_label in enumerate(nb.classes_):
    # Get mean and variance for the current class and feature
    mean = nb.theta_[i, feature_index]
    var = np.sqrt(nb.var_[i, feature_index])
    
    # Compute the Gaussian PDF for the range of x values
    pdf = norm.pdf(x_axis, mean, np.sqrt(var))
    
    # Plot the distribution
    plt.plot(x_axis, pdf, color=label_colors[class_label]/255.,  label=f'Class {class_label} (mean={mean:.2f}, var={var:.2f})')

# Add labels and legend
plt.title(f'Gaussian Distribution for Feature {feature_index}')
plt.xlabel(f'Feature {feature_index}')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# %%
    
evaluate_classification(nb_model, 'GaussianNB', test_preds=y_pred_knn) # Takes a minute to run

# %% [markdown]
# ### Supervised Classification: Decision Tree
#
# **Overview:**
# - Tree-based model that splits data based on feature values
# - Easy to interpret

# %%
from sklearn.tree import DecisionTreeClassifier
# Initialize Decision Tree

dt = DecisionTreeClassifier(random_state=42, max_depth=15)

dt_model = make_pipeline(scaler, dt, verbose=True)

# Fit Decision Tree
dt_model.fit(X_train, y_train, decisiontreeclassifier__sample_weight=sample_weights)

# %%
from sklearn.tree import plot_tree

# Plot the decision tree
plt.figure(figsize=(12,8))  # Adjust the figure size as needed
plot_tree(dt, filled=True, rounded=True, max_depth=2, 
          feature_names=['R', 'G', 'B', 'NiR'], 
          class_names=label_names)
plt.show()

# %%

evaluate_classification(dt_model, 'Decision Tree') 

# %% [markdown]
# ### Supervised Classification: Random Forest
#
# **Overview:**
# - Ensemble of decision trees
# - Reduces overfitting and improves accuracy

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

X_train_scaled = scaler.transform(X_train)  # Avoid refitting scaler

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train, sample_weight=sample_weights)

rf_model = Pipeline([('scaler', scaler), ('rf', rf)])

evaluate_classification(rf_model, 'Random Forest') 

# %% [markdown]
# ### Supervised Classification: Gradient Boosting
#
# **Overview:**
# - Ensemble method that builds trees sequentially
# - Each tree corrects errors of the previous ones

# %%
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=15, learning_rate=0.1, random_state=42)

gb.fit(X_train_scaled, y_train, sample_weight=sample_weights)

gb_model = Pipeline([('scaler', scaler), ('gb', gb)])

# %%
evaluate_classification(gb_model, 'Gradient Boosting') 

# %% [markdown]
# ### Supervised Classification: Multi-Layer Perceptron (MLP) Classifier
#
# **Overview:**
# - Feedforward neural network
# - Can capture complex relationships

# %%
from sklearn.neural_network import MLPClassifier
# Initialize MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10),  max_iter=30, random_state=42)
mlp.fit(X_train, y_train)

# No sample weights or class weights for the built-in  MLP in sklearn

mlp_model = Pipeline([('scaler', scaler), ('mlp', mlp)])


# %%

evaluate_classification(mlp_model, 'MLP') 

# %%
