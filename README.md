# LULC Classification Using Random Forest in Google Earth Engine (GEE)

Code for the paper *"How does air quality reflect land cover changes: A remote sensing approach using Sentinel data."* This repository includes data processing, analysis, and visualization scripts used to examine the relationship between air quality and land cover dynamics in southern Ecuador.

## Project Overview

This project leverages **Sentinel-1**, **Sentinel-2**, and **Sentinel-5P** data to classify land cover and analyze air quality. The primary objectives are:
- To perform Land Use and Land Cover (LULC) classification using **Random Forest (RF)** in **Google Earth Engine**.
- To examine the relationship between LULC classes and air pollutants (O₃, SO₂, NO₂) using data from Sentinel-5P.

## Datasets Used

- **Sentinel-1 (SAR)**: Provides radar imagery, effective for distinguishing structural features of land cover types.
- **Sentinel-2 (Optical)**: Offers high-resolution multispectral imagery, useful for detailed land cover classification.
- **Sentinel-5P (Atmospheric)**: Provides pollutant concentration data for O₃, SO₂, and NO₂, aiding in air quality analysis.

## Methodology

1. **Data Collection & Preprocessing**:
   - Sentinel-1 and Sentinel-2 data were preprocessed (e.g., noise reduction and cloud masking) to enhance data quality.
   - Sentinel-5P data was resampled to match the spatial resolution of the LULC dataset to facilitate spatial analysis.

2. **Classification**:
   - The **Random Forest algorithm** was used to classify land cover types with separate models for Sentinel-1, Sentinel-2, and combined datasets.
   - Classification was performed on GEE, allowing for efficient processing of large datasets.

3. **Air Quality Analysis**:
   - Pollution levels of O₃, SO₂, and NO₂ from Sentinel-5P were analyzed across different land cover types to explore potential relationships between air quality and LULC.

## How to Use

1. **Run the code** in Google Earth Engine using the `lulc_classification_RF.js` script.
2. **Adjust the study area** geometry if needed.
3. **Execute the script** to generate LULC maps and export them for further analysis.

## Requirements

- **Google Earth Engine account**
- **Basic knowledge of JavaScript and GEE**

## Acknowledgments

This work was funded by the Estonian Research Agency (grant number PRG1764), the Estonian Ministry of Education and Research, and the Centre of Excellence for Sustainable Land Use (TK232).
