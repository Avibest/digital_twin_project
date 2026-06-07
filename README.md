# Digital Twin for Bearing Fault Detection

A data-driven digital twin for early fault detection in rotating 
machinery, built using vibration data from the NASA IMS bearing dataset.

## What it does
Extracts four physical health indicators (RMS, kurtosis, crest factor, 
spectral energy) from raw accelerometer signals, trains a Random Forest A 
classifier to distinguish healthy from failing bearings, and visualizes 
degradation trends through an interactive Streamlit interface.

## Key results
- 100% classification accuracy on a binary healthy/failure task
- RMS and spectral energy identified as dominant predictive features
- Health score tracks bearing degradation in real time

## Limitations
- Binary classification only, no remaining useful life (RUL) estimation
- Single accelerometer channel per file
- Trained on reduced dataset subset for deployment

## How to run
pip install -r requirements.txt
Unzip data_small.zip before running.
streamlit run app_small.py

## Data
Full NASA IMS bearing dataset available at:
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
