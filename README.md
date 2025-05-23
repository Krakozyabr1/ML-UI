# Web Application for Electrophysiological Signal Analysis

This web application is designed for the analysis of electrophysiological signals, specifically focusing on electroencephalogram (EEG), electrocardiogram (ECG), and cardiorhythmogram (CRG) signals. The software automates many routine operations, including data processing and model training.
Currently, it supports multi-channel recordings in the .edf format.
Also it can be used with tabular data (in this case just skip feature extraction step and put csv file into Features folder).

**Key Features:**

* **Signal Visualization:** 
    * Visual analysis of signals, including:
        * Signal plots
        * Spectrum plots
        * Spectrogram plots
        * Average power spectral density plots
    * Customizable parameters:
        * Time interval
        * Frequency range
        * Window width
        * Overlap percentage
    * Logarithmic scale options for spectrum, spectrogram, and average power spectral density. 

* **CRG Signal Extraction:** 
    * Extracts CRG signals from ECG signals.
    * Generates .edf files containing CRG and interpolated CRG signals (8 Hz sampling frequency).

* **Feature Extraction:** 
    * Extracts features from EEG, ECG, and CRG signals.
    * Supports training and classification modes:
        * **Training:** Generates a feature table for model training with class labels.
        * **Classification:** Generates a feature table for signals to be classified.
    * Saves feature tables as .csv files in the "Features/Learning" folder.

* **Feature Importance Analysis:** 
    * Analyzes feature importance using methods like F-measure, mutual information, and Spearman correlation coefficient.
    * Visualizes feature importance with column charts.
    * Provides options for visualizing feature distributions (histograms, box plots) and scatter diagrams.

* **Model Pre-tuning:** 
    * Pre-tunes machine learning (ML) models for feature selection using hierarchical search.
    * Supports selecting the number of features for hierarchical search to speed up tuning.
    * Displays cross-validation accuracy (or r2 for regression), test accuracy, and discrepancy matrices for each model.
    * Allows users to select and save models for further training as .pkl files.

* **Feature Selection:** 
    * Selects features using sequential feature selection methods or feature importances given by model itself.
    * Supports selecting the maximum number of features and the size of the initial feature subset.
    * Uses different feature selection methods based on the ML model (e.g., F-measure for support vector machines).
    * Saves selected models and feature lists as .pkl files.

* **Model Tuning:** 
    * Tunes selected models based on feature selection results.
    * Requires a .csv file with features and a .pkl file from the previous feature selection step.

* **Data Analysis:** 
    * Analyses (classification or regression) new data using trained models.
    * Requires a .csv file with features and a .pkl file with trained models.
    * Outputs a .csv file with models outputs.

**Installation:**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, use the provided install_reqs.bat script.
2. **Run the application:**
   * Open a command window in the application folder.
   * Run the command:
   ```bash
   streamlit run "ML UI.py"
   ```
   You can also use the run.bat script (might be more convenient on Windows).
   
