# Sentinel-2 Feature Mask Extraction with Python

This repository contains a Python workflow for extracting and visualizing **feature masks** from Sentinel-2 imagery using commonly used spectral indices.

## Features
- **Band processing** with [Rasterio](https://rasterio.readthedocs.io/).  
- **Index calculation** with NumPy.  
- **Feature masks** for vegetation, water, built-up areas, soil/vegetation, moisture, and burned areas.  
- **Visualization** of composites, indices, and masks with Matplotlib.  

## Spectral Indices Implemented
- **NDVI** – Vegetation  
- **NDWI** – Water  
- **NDBI** – Built-up  
- **NDMI** – Moisture  
- **SAVI** – Soil-Adjusted Vegetation Index  
- **BAI** – Burn Area Index  

## Dependencies
- Python 3.8+  
- [Rasterio](https://rasterio.readthedocs.io/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- OS & Glob (standard Python libraries)  

Install dependencies with:
```bash
pip install rasterio numpy matplotlib
