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

# %% [markdown] {"id": "hyper-title", "slideshow": {"slide_type": "slide"}}
# # Spectral Mixtures and Matching
#
# [Open this notebook in Colab](https://colab.research.google.com/github/jfemiani/cse620b-public/blob/main/Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.ipynb)
#
# A self-contained, synthetic example of hyperspectral reasoning. The curves
# below are schematic, not field measurements or spectral-library records.

# %% [markdown] {"id": "hyper-spectra", "slideshow": {"slide_type": "slide"}}
# ## Build two schematic material spectra
#
# A hyperspectral pixel is a vector with one value per wavelength. Here we use
# smooth curves to represent a vegetation-like spectrum and a soil-like
# spectrum. The goal is to study the operations, not identify real materials.

# %% {"id": "hyper-imports", "slideshow": {"slide_type": "skip"}}
import matplotlib.pyplot as plt
import numpy as np

# %% {"id": "hyper-generate", "slideshow": {"slide_type": "slide"}}
wavelength_nm = np.linspace(400, 1000, 121)
red_absorption = 0.07 * np.exp(-0.5 * ((wavelength_nm - 670) / 22) ** 2)
green_peak = 0.07 * np.exp(-0.5 * ((wavelength_nm - 550) / 35) ** 2)
red_edge = 0.42 / (1 + np.exp(-(wavelength_nm - 720) / 14))
vegetation = 0.06 + green_peak - red_absorption + red_edge
soil = 0.12 + 0.00028 * (wavelength_nm - 400) + 0.012 * np.sin(wavelength_nm / 90)

fig, axis = plt.subplots(figsize=(8, 4))
axis.plot(wavelength_nm, vegetation, label="Vegetation-like (synthetic)")
axis.plot(wavelength_nm, soil, label="Soil-like (synthetic)")
axis.set(xlabel="Wavelength (nm)", ylabel="Illustrative reflectance", title="Two schematic endmembers")
axis.legend()
axis.grid(alpha=0.2)
plt.show()

# %% [markdown] {"id": "hyper-mixture", "slideshow": {"slide_type": "slide"}}
# ## Mix spectra and estimate fractions
#
# Under the linear model, the observed spectrum is a weighted sum of endmember
# spectra. With two endmembers, fit one fraction along the line from soil to
# vegetation, restrict it to [0, 1], and use its complement for soil. This
# enforces nonnegativity and a fraction sum of one. Real analyses still need
# good endmembers and residual checks.

# %% {"id": "hyper-mixture-code", "slideshow": {"slide_type": "slide"}}
true_fractions = np.array([0.65, 0.35])
endmembers = np.column_stack((vegetation, soil))
pixel = endmembers @ true_fractions

spectral_direction = vegetation - soil
vegetation_fraction = np.clip(
    np.dot(pixel - soil, spectral_direction) / np.dot(spectral_direction, spectral_direction),
    0, 1,
)
estimated_fractions = np.array([vegetation_fraction, 1 - vegetation_fraction])
residual = pixel - endmembers @ estimated_fractions

print(f"True fractions:      vegetation={true_fractions[0]:.2f}, soil={true_fractions[1]:.2f}")
print(f"Least-squares fit:   vegetation={estimated_fractions[0]:.2f}, soil={estimated_fractions[1]:.2f}")
print(f"Fraction sum: {estimated_fractions.sum():.3f}; RMS residual: {np.sqrt(np.mean(residual**2)):.5f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(wavelength_nm, pixel, color="black", label="Mixed pixel")
axes[0].plot(wavelength_nm, endmembers @ estimated_fractions, "--", label="Reconstruction")
axes[0].set(xlabel="Wavelength (nm)", ylabel="Illustrative reflectance", title="Observed and reconstructed")
axes[0].legend()
axes[1].bar(["Vegetation", "Soil"], estimated_fractions, color=["forestgreen", "sandybrown"])
axes[1].set(ylim=(0, 1), ylabel="Estimated fraction", title="Estimated mixture")
plt.tight_layout()
plt.show()

# %% [markdown] {"id": "hyper-angle", "slideshow": {"slide_type": "slide"}}
# ## Spectral angle is not abundance
#
# Spectral Angle Mapping compares spectral shape. Multiplying a spectrum by a
# positive brightness factor leaves its angle unchanged, while unmixing asks
# for component proportions under a mixing model.

# %% {"id": "hyper-angle-code", "slideshow": {"slide_type": "slide"}}
def spectral_angle(first, second):
    cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

dim_vegetation = 0.45 * vegetation
print(f"Angle to vegetation reference: {spectral_angle(dim_vegetation, vegetation):.6f} degrees")
print(f"Angle to soil reference:        {spectral_angle(dim_vegetation, soil):.2f} degrees")

# %% [markdown] {"id": "hyper-question", "slideshow": {"slide_type": "slide"}}
# ## Check the assumptions
#
# - What changed when the vegetation spectrum was dimmed? What did not?
# - Why can a small angle to a library spectrum fail to identify a pure pixel?
# - What would you inspect if a real pixel had a large reconstruction residual?
#
# The curves are synthetic and deliberately simple. Use calibrated scene
# reflectance and documented reference spectra for material interpretation.