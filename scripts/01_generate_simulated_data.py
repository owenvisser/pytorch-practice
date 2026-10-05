"""
Generate Complete Simulated Longitudinal Data
=============================================

This script generates the complete simulated longitudinal data used in the
longitudinal graph neural network experiments.

The underlying data-generating mechanism is defined in:

    src/data_simulation.py

Two data sets are created:

1. complete_longitudinal_data.csv
   One row per patient visit containing irregular observation times and
   five complete longitudinal measurements.

2. patient_data.csv
   One row per patient containing the patient-level latent variables,
   binary classification outcome, and continuous regression outcome.

Missingness is NOT introduced in this script. The complete measurements are
saved first so that they can later serve as ground truth when evaluating
missing-data and imputation methods.

Run this script from the root PyTorch-Practice directory using:

    python scripts/01_generate_simulated_data.py
"""

from pathlib import Path
import sys
import pandas as pd

# Find the root PyTorch-Practice directory from the location of this script.
# Add the project root to Python's search path so that modules inside src/
# can be imported when this script is executed from the scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from src.data_simulation import simulate_longitudinal_data

# SIMULATION SETTINGS
N_PATIENTS = 5000
MIN_VISITS = 5
MAX_VISITS = 20
REGRESSION_NOISE_SD = 1
RANDOM_SEED = 100

# OUTPUT DIRECTORY
SIMULATED_DATA_DIRECTORY = (PROJECT_ROOT/"data"/"simulated")
SIMULATED_DATA_DIRECTORY.mkdir(parents=True,exist_ok=True)

# GENERATE COMPLETE LONGITUDINAL DATA
longitudinal_data, patient_data = simulate_longitudinal_data(
    n_patients=N_PATIENTS,
    min_visits=MIN_VISITS,
    max_visits=MAX_VISITS,
    regression_noise_sd=REGRESSION_NOISE_SD,
    seed=RANDOM_SEED
)

# DISPLAY OUTCOME SUMMARY
classification_counts = (patient_data["classification_outcome"].value_counts().sort_index())
classification_proportions = (patient_data["classification_outcome"].value_counts(normalize=True).sort_index())

print(pd.DataFrame({
        "count": classification_counts,
        "proportion": classification_proportions
    }))

print(patient_data["regression_outcome"].describe())
print(longitudinal_data.head())
print(patient_data.head())

# SAVE COMPLETE SIMULATED DATA
longitudinal_output_path = (SIMULATED_DATA_DIRECTORY/"complete_longitudinal_data.csv")
longitudinal_data.to_csv(longitudinal_output_path,index=False)
print("Longitudinal data saved to:",longitudinal_output_path)

patient_output_path = (SIMULATED_DATA_DIRECTORY/"patient_data.csv")
patient_data.to_csv(patient_output_path,index=False)
print("Patient data saved to:",patient_output_path)