"""
Prepare Data Sets for LongGCN
=============================

This script converts each train, validation, and test longitudinal data set
into the observation-level format expected by the LongGCN package.

The visit-level simulated data contain one row per patient visit:

    patient_id
    visit_number
    time
    x1
    x2
    x3
    x4
    x5

The LongGCN representation contains one row per observed scalar measurement:

    patient_id
    time
    measurement
    value

Missing measurements are omitted from this representation.

Four data conditions are prepared:

1. Complete data
2. MCAR-only data
3. Group-specific masked data
4. Group-specific masked data with additional MCAR missingness

Two measurement-group structures are used:

Single-group structure
    {x1, x2, x3, x4, x5}

Two-group structure
    Group 1 = {x1, x2, x3}
    Group 2 = {x4, x5}

The single-group structure is used for complete and MCAR-only data.
The two-group structure is used for the group-specific data conditions.

The resulting observation-level CSV files are saved for reproducibility.
LongGCN objects themselves are constructed from these files as needed rather
than being serialized to disk.

Run this script from the root PyTorch-Practice directory using:

    python scripts/04_prepare_longgcn_datasets.py
"""


from pathlib import Path
import json
import sys

import pandas as pd

from longgcn.data import (
    LongitudinalDataset,
    DesignedMeasurementGroups
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from src.longgcn_data_preparation import convert_to_longgcn_format


# =============================================================================
# 1. LONGITUDINAL MEASUREMENT DEFINITIONS
# =============================================================================

MEASUREMENT_COLUMNS = [
    "x1",
    "x2",
    "x3",
    "x4",
    "x5"
]


# All measurements belong to one common group.

SINGLE_GROUP_DEFINITION = {
    "all_measurements": [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5"
    ]
}


# Measurements belong to two distinct designed groups.

TWO_GROUP_DEFINITION = {
    "group_1": [
        "x1",
        "x2",
        "x3"
    ],
    "group_2": [
        "x4",
        "x5"
    ]
}


# =============================================================================
# 2. INPUT AND OUTPUT DIRECTORIES
# =============================================================================

SPLIT_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


LONGGCN_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "longgcn"
)


LONGGCN_DATA_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 3. DEFINE DATA CONDITIONS
# =============================================================================

# Each data condition records:
#
#     1. the filename pattern used in data/splits/
#     2. the measurement-group structure used by LongGCN

DATA_CONDITIONS = {

    "complete": {
        "filename_suffix": "complete_longitudinal_data.csv",
        "groups": SINGLE_GROUP_DEFINITION
    },

    "mcar": {
        "filename_suffix": "mcar_longitudinal_data.csv",
        "groups": SINGLE_GROUP_DEFINITION
    },

    "group_specific": {
        "filename_suffix": "group_specific_longitudinal_data.csv",
        "groups": TWO_GROUP_DEFINITION
    },

    "group_specific_mcar": {
        "filename_suffix": "group_specific_mcar_longitudinal_data.csv",
        "groups": TWO_GROUP_DEFINITION
    }
}


DATA_SPLITS = [
    "train",
    "validation",
    "test"
]


# =============================================================================
# 4. CREATE LONGGCN MEASUREMENT-GROUP OBJECTS
# =============================================================================

single_group_design = DesignedMeasurementGroups(
    measurements=MEASUREMENT_COLUMNS,
    groups=SINGLE_GROUP_DEFINITION
)


two_group_design = DesignedMeasurementGroups(
    measurements=MEASUREMENT_COLUMNS,
    groups=TWO_GROUP_DEFINITION
)


# =============================================================================
# 5. CONVERT EACH DATA SET TO LONGGCN FORMAT
# =============================================================================

# Keep the constructed LongitudinalDataset objects in memory while the script
# runs so that we can verify every data condition successfully passes through
# the LongGCN package.

longgcn_datasets = {}


print()
print("============================================================")
print("PREPARING LONGGCN DATA SETS")
print("============================================================")


for condition_name, condition_settings in DATA_CONDITIONS.items():

    print()
    print(
        condition_name.upper()
    )
    print(
        "-" * len(condition_name)
    )


    # Create a separate output directory for this data condition.

    condition_output_directory = (
        LONGGCN_DATA_DIRECTORY
        / condition_name
    )


    condition_output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    longgcn_datasets[
        condition_name
    ] = {}


    for split_name in DATA_SPLITS:

        # ---------------------------------------------------------------------
        # LOAD VISIT-LEVEL DATA
        # ---------------------------------------------------------------------

        input_filename = (
            f"{split_name}_"
            f"{condition_settings['filename_suffix']}"
        )


        input_path = (
            SPLIT_DATA_DIRECTORY
            / input_filename
        )


        visit_level_data = pd.read_csv(
            input_path
        )


        # ---------------------------------------------------------------------
        # CONVERT TO OBSERVATION-LEVEL LONGGCN FORMAT
        # ---------------------------------------------------------------------

        observation_level_data = convert_to_longgcn_format(
            longitudinal_data=visit_level_data,
            measurement_columns=MEASUREMENT_COLUMNS
        )


        # ---------------------------------------------------------------------
        # CONSTRUCT LONGGCN DATA SET
        # ---------------------------------------------------------------------

        # LongitudinalDataset uses only the patient, time, measurement, and
        # value columns when constructing the generalized patient-level
        # representation.

        dataset = LongitudinalDataset(
            data=observation_level_data,
            patient_col="patient_id",
            time_col="time",
            measurement_col="measurement",
            value_col="value"
        )


        longgcn_datasets[
            condition_name
        ][
            split_name
        ] = dataset


        # ---------------------------------------------------------------------
        # SAVE OBSERVATION-LEVEL DATA
        # ---------------------------------------------------------------------

        output_path = (
            condition_output_directory
            / f"{split_name}_longgcn_data.csv"
        )


        observation_level_data.to_csv(
            output_path,
            index=False
        )


        # ---------------------------------------------------------------------
        # DISPLAY SUMMARY
        # ---------------------------------------------------------------------

        print(
            f"{split_name}: "
            f"{len(dataset)} patients, "
            f"{len(observation_level_data)} observed measurements"
        )


# =============================================================================
# 6. SAVE MEASUREMENT-GROUP DEFINITIONS
# =============================================================================

group_definition_output = {
    "single_group": SINGLE_GROUP_DEFINITION,
    "two_group": TWO_GROUP_DEFINITION
}


with open(
    LONGGCN_DATA_DIRECTORY
    / "measurement_group_definitions.json",
    "w"
) as file:

    json.dump(
        group_definition_output,
        file,
        indent=4
    )


# =============================================================================
# 7. INSPECT ONE COMPLETE-DATA PATIENT
# =============================================================================

complete_training_dataset = (
    longgcn_datasets[
        "complete"
    ][
        "train"
    ]
)


# Use the first training patient listed in the saved split file.

training_patient_data = pd.read_csv(
    SPLIT_DATA_DIRECTORY
    / "train_patient_data.csv"
)


example_patient_id = int(
    training_patient_data[
        "patient_id"
    ].iloc[0]
)


complete_patient = complete_training_dataset[
    example_patient_id
]


complete_group_data = single_group_design.transform(
    complete_patient
)


print()
print("============================================================")
print("EXAMPLE COMPLETE-DATA PATIENT")
print("============================================================")

print(
    "Patient ID:",
    example_patient_id
)

print(
    "Number of observation times:",
    len(complete_patient.times)
)

print(
    "X shape:",
    tuple(
        complete_patient.X.shape
    )
)

print(
    "Group membership matrix Q shape:",
    tuple(
        complete_group_data.Q.shape
    )
)

print(
    "Number of designed groups:",
    single_group_design.G
)


# =============================================================================
# 8. INSPECT ONE GROUP-SPECIFIC PATIENT
# =============================================================================

group_training_dataset = (
    longgcn_datasets[
        "group_specific"
    ][
        "train"
    ]
)


group_patient = group_training_dataset[
    example_patient_id
]


group_data = two_group_design.transform(
    group_patient
)


print()
print("============================================================")
print("EXAMPLE GROUP-SPECIFIC PATIENT")
print("============================================================")

print(
    "Patient ID:",
    example_patient_id
)

print(
    "Number of observation times:",
    len(group_patient.times)
)

print(
    "X shape:",
    tuple(
        group_patient.X.shape
    )
)

print(
    "Group membership matrix Q shape:",
    tuple(
        group_data.Q.shape
    )
)

print(
    "Number of designed groups:",
    two_group_design.G
)


# =============================================================================
# 9. CONFIRM OUTPUT
# =============================================================================

print()
print("============================================================")
print("LONGGCN DATA PREPARATION COMPLETE")
print("============================================================")

print()
print(
    "Prepared data saved to:",
    LONGGCN_DATA_DIRECTORY
)

print()