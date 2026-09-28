"""
Create Patient-Level Train, Validation, and Test Splits
=======================================================

This script creates the master patient-level train, validation, and test
assignments used throughout the longitudinal graph neural network simulation
study.

The patient split is generated ONCE and then applied to every version of the
longitudinal data. This guarantees that all model comparisons use exactly the
same patients for training, validation, and testing.

The following data sets are split:

1. Complete longitudinal data
2. MCAR-only longitudinal data
3. Group-specific masked longitudinal data
4. Group-specific masked longitudinal data with additional MCAR missingness
5. Patient-level data containing outcomes and latent simulation variables

The patient split assignments are also saved separately so that the same
partition can later be reused for imputed data and additional model variants.

Run this script from the root PyTorch-Practice directory using:

    python scripts/03_create_data_splits.py
"""


from pathlib import Path
import sys

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from src.data_splitting import (
    create_patient_split_assignments,
    split_longitudinal_data,
    split_patient_data
)


# =============================================================================
# 1. SPLIT SETTINGS
# =============================================================================

TRAIN_PROPORTION = 0.70

VALIDATION_PROPORTION = 0.15

TEST_PROPORTION = 0.15

RANDOM_SEED = 100


# =============================================================================
# 2. INPUT DIRECTORIES
# =============================================================================

SIMULATED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "simulated"
)


INCOMPLETE_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "incomplete"
)


# =============================================================================
# 3. OUTPUT DIRECTORY
# =============================================================================

SPLIT_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


SPLIT_DATA_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 4. LOAD PATIENT-LEVEL DATA
# =============================================================================

patient_data = pd.read_csv(
    SIMULATED_DATA_DIRECTORY
    / "patient_data.csv"
)


# =============================================================================
# 5. CREATE MASTER PATIENT SPLIT ASSIGNMENTS
# =============================================================================

split_assignments = create_patient_split_assignments(
    patient_data=patient_data,
    train_proportion=TRAIN_PROPORTION,
    validation_proportion=VALIDATION_PROPORTION,
    test_proportion=TEST_PROPORTION,
    seed=RANDOM_SEED
)


# =============================================================================
# 6. DISPLAY SPLIT COUNTS
# =============================================================================

split_counts = (
    split_assignments[
        "data_split"
    ]
    .value_counts()
    .reindex(
        [
            "train",
            "validation",
            "test"
        ]
    )
)


split_proportions = (
    split_counts
    / len(split_assignments)
)


split_summary = pd.DataFrame({
    "number_of_patients": split_counts,
    "proportion": split_proportions
})


print()
print("============================================================")
print("PATIENT-LEVEL DATA SPLITS")
print("============================================================")

print()
print(
    split_summary
)


# =============================================================================
# 7. SPLIT PATIENT-LEVEL DATA
# =============================================================================

(
    training_patient_data,
    validation_patient_data,
    test_patient_data

) = split_patient_data(

    patient_data=patient_data,
    split_assignments=split_assignments
)


# =============================================================================
# 8. DISPLAY OUTCOME DISTRIBUTIONS
# =============================================================================

print()
print("CLASSIFICATION OUTCOME BY DATA SPLIT")
print("------------------------------------")


for split_name, split_data in [

    (
        "Training",
        training_patient_data
    ),

    (
        "Validation",
        validation_patient_data
    ),

    (
        "Test",
        test_patient_data
    )

]:

    positive_proportion = (
        split_data[
            "classification_outcome"
        ]
        .mean()
    )


    print(
        f"{split_name}: "
        f"{positive_proportion:.4f}"
    )


print()
print("REGRESSION OUTCOME BY DATA SPLIT")
print("--------------------------------")


for split_name, split_data in [

    (
        "Training",
        training_patient_data
    ),

    (
        "Validation",
        validation_patient_data
    ),

    (
        "Test",
        test_patient_data
    )

]:

    regression_mean = (
        split_data[
            "regression_outcome"
        ]
        .mean()
    )


    regression_sd = (
        split_data[
            "regression_outcome"
        ]
        .std()
    )


    print(
        f"{split_name}: "
        f"mean = {regression_mean:.4f}, "
        f"SD = {regression_sd:.4f}"
    )


# =============================================================================
# 9. LOAD ALL LONGITUDINAL DATA SETS
# =============================================================================

complete_longitudinal_data = pd.read_csv(
    SIMULATED_DATA_DIRECTORY
    / "complete_longitudinal_data.csv"
)


mcar_longitudinal_data = pd.read_csv(
    INCOMPLETE_DATA_DIRECTORY
    / "mcar_longitudinal_data.csv"
)


group_specific_longitudinal_data = pd.read_csv(
    INCOMPLETE_DATA_DIRECTORY
    / "group_specific_longitudinal_data.csv"
)


group_specific_mcar_longitudinal_data = pd.read_csv(
    INCOMPLETE_DATA_DIRECTORY
    / "group_specific_mcar_longitudinal_data.csv"
)


# =============================================================================
# 10. SPLIT COMPLETE LONGITUDINAL DATA
# =============================================================================

(
    complete_training_data,
    complete_validation_data,
    complete_test_data

) = split_longitudinal_data(

    longitudinal_data=complete_longitudinal_data,
    split_assignments=split_assignments
)


# =============================================================================
# 11. SPLIT MCAR-ONLY LONGITUDINAL DATA
# =============================================================================

(
    mcar_training_data,
    mcar_validation_data,
    mcar_test_data

) = split_longitudinal_data(

    longitudinal_data=mcar_longitudinal_data,
    split_assignments=split_assignments
)


# =============================================================================
# 12. SPLIT GROUP-SPECIFIC LONGITUDINAL DATA
# =============================================================================

(
    group_training_data,
    group_validation_data,
    group_test_data

) = split_longitudinal_data(

    longitudinal_data=group_specific_longitudinal_data,
    split_assignments=split_assignments
)


# =============================================================================
# 13. SPLIT GROUP-SPECIFIC + MCAR LONGITUDINAL DATA
# =============================================================================

(
    group_mcar_training_data,
    group_mcar_validation_data,
    group_mcar_test_data

) = split_longitudinal_data(

    longitudinal_data=group_specific_mcar_longitudinal_data,
    split_assignments=split_assignments
)


# =============================================================================
# 14. SAVE MASTER PATIENT SPLIT ASSIGNMENTS
# =============================================================================

split_assignments.to_csv(
    SPLIT_DATA_DIRECTORY
    / "patient_split_assignments.csv",
    index=False
)


# =============================================================================
# 15. SAVE PATIENT-LEVEL DATA SPLITS
# =============================================================================

training_patient_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "train_patient_data.csv",
    index=False
)


validation_patient_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "validation_patient_data.csv",
    index=False
)


test_patient_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "test_patient_data.csv",
    index=False
)


# =============================================================================
# 16. SAVE COMPLETE LONGITUDINAL DATA SPLITS
# =============================================================================

complete_training_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "train_complete_longitudinal_data.csv",
    index=False
)


complete_validation_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "validation_complete_longitudinal_data.csv",
    index=False
)


complete_test_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "test_complete_longitudinal_data.csv",
    index=False
)


# =============================================================================
# 17. SAVE MCAR-ONLY LONGITUDINAL DATA SPLITS
# =============================================================================

mcar_training_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "train_mcar_longitudinal_data.csv",
    index=False
)


mcar_validation_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "validation_mcar_longitudinal_data.csv",
    index=False
)


mcar_test_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "test_mcar_longitudinal_data.csv",
    index=False
)


# =============================================================================
# 18. SAVE GROUP-SPECIFIC LONGITUDINAL DATA SPLITS
# =============================================================================

group_training_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "train_group_specific_longitudinal_data.csv",
    index=False
)


group_validation_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "validation_group_specific_longitudinal_data.csv",
    index=False
)


group_test_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "test_group_specific_longitudinal_data.csv",
    index=False
)


# =============================================================================
# 19. SAVE GROUP-SPECIFIC + MCAR LONGITUDINAL DATA SPLITS
# =============================================================================

group_mcar_training_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "train_group_specific_mcar_longitudinal_data.csv",
    index=False
)


group_mcar_validation_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "validation_group_specific_mcar_longitudinal_data.csv",
    index=False
)


group_mcar_test_data.to_csv(
    SPLIT_DATA_DIRECTORY
    / "test_group_specific_mcar_longitudinal_data.csv",
    index=False
)


# =============================================================================
# 20. CONFIRM SAVED FILES
# =============================================================================

print()
print("============================================================")
print("DATA SPLITTING COMPLETE")
print("============================================================")

print()
print(
    "Split data saved to:",
    SPLIT_DATA_DIRECTORY
)

print()