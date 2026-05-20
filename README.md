# BirdSound3D

A Python project for processing bird call audio, extracting features, creating 3D visualizations, and classifying bird species.
 
## Features

- Data Collection: Load bird call audio files from local folders or Xeno-canto API
- Preprocessing: Normalize audio, reduce noise, and trim silence
- Feature Extraction: Compute MFCCs, spectrograms, and chromagrams using Librosa
- 3D Visualization: Create interactive 3D visualizations of audio features
- Classification: Train ML models to identify bird species
- Interface: Simple Streamlit app for uploading and processing audio files

## Installation

```bash
pip install -r requirements.txt 
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `data_collection.py`: Functions for loading and organizing audio data
- `preprocessing.py`: Audio preprocessing functions
- `feature_extraction.py`: Feature extraction using Librosa
- `visualization.py`: 3D visualization functions
- `classification.py`: ML model training and prediction
- `utils.py`: Utility functions
