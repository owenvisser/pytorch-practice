"""
Prepare LongGCN Datasets
========================

This script converts the previously created train, validation, and test data
splits into model-ready LongGCN PyTorch datasets.

The script assumes that:

    scripts/01_generate_simulated_data.py
    scripts/02_generate_incomplete_datasets.py
    scripts/03_create_data_splits.py

have already been run.

Four longitudinal-data conditions are prepared:

    complete
        Fully observed longitudinal data.

    mcar
        Individual measurements removed completely at random.

    group_specific
        Observation times follow the predefined measurement-group structure.

    group_specific_mcar
        Group-specific observation structure with additional MCAR missingness.

Two prediction tasks are prepared for every condition:

    classification
        Uses classification_outcome.

    regression
        Uses regression_outcome.

The complete and MCAR datasets use one designed measurement group containing
all five measurements.

The group-specific datasets use two designed groups:

    group_1
        x1, x2, x3

    group_2
        x4, x5

For each condition, task, and data split, the script constructs and saves a
LongGCNTorchDataset with precompute=True.

The saved datasets therefore already contain the patient-specific:

    X_i
    T_i
    P_i
    A_i
    y_i

representations required for model training.

Run from the project root using:

    python scripts/04_prepare_longgcn_datasets.py
"""


from pathlib import Path
import json
import sys

import pandas as pd
import torch


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# Allow imports from the local src directory when the script is run directly
# from the project root.

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.longgcn_data_preparation import (
    convert_to_longgcn_format
)


from longgcn.data import (
    LongitudinalDataset,
    DesignedMeasurementGroups,
    LongGCNTorchDataset
)


# =============================================================================
# 2. INPUT AND OUTPUT DIRECTORIES
# =============================================================================

SPLIT_DIRECTORY = (
    PROJECT_ROOT
    /
    "data"
    /
    "splits"
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    /
    "data"
    /
    "longgcn"
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 3. MEASUREMENT SETTINGS
# =============================================================================

MEASUREMENT_COLUMNS = [
    "x1",
    "x2",
    "x3",
    "x4",
    "x5"
]


# =============================================================================
# 4. TEMPORAL WEIGHTING SETTINGS
# =============================================================================

# Temporal communication uses:
#
#     exp(-delta_t / d)
#
# The value used in the initial simulation experiments is specified here.
#
# Because T_i and A_ig are precomputed, changing this value requires rerunning
# this preparation script.

DECAY_PARAMETER = 0.75


# =============================================================================
# 5. PRECOMPUTATION SETTING
# =============================================================================

# Precomputing avoids reconstructing T_i, P_i, and A_ig during every training
# epoch.

PRECOMPUTE = True


# =============================================================================
# 6. DESIGNED MEASUREMENT GROUPS
# =============================================================================

SINGLE_GROUP_DEFINITION = {

    "all_measurements": [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5"
    ]
}


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
# 7. LONGITUDINAL DATA CONDITIONS
# =============================================================================

DATA_CONDITIONS = {

    "complete": {
        "file_label":
            "complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },

    "mcar": {
        "file_label":
            "mcar",

        "groups":
            SINGLE_GROUP_DEFINITION
    },

    "group_specific": {
        "file_label":
            "group_specific",

        "groups":
            TWO_GROUP_DEFINITION
    },

    "group_specific_mcar": {
        "file_label":
            "group_specific_mcar",

        "groups":
            TWO_GROUP_DEFINITION
    }
}


# =============================================================================
# 8. PREDICTION TASKS
# =============================================================================

PREDICTION_TASKS = {

    "classification":
        "classification_outcome",

    "regression":
        "regression_outcome"
}


# =============================================================================
# 9. DATA SPLITS
# =============================================================================

DATA_SPLITS = [
    "train",
    "validation",
    "test"
]


# =============================================================================
# CREATE PATIENT-LEVEL OUTCOME MAPPING
# =============================================================================

def create_outcome_mapping(
    patient_data,
    outcome_column
):
    """
    Create a patient-ID-to-outcome mapping.

    Parameters
    ----------
    patient_data : pandas.DataFrame
        Patient-level data containing patient_id and the requested outcome.

    outcome_column : str
        Name of the patient-level outcome column.

    Returns
    -------
    outcomes : dict
        Mapping:

            patient_id -> outcome
    """

    outcomes = dict(
        zip(
            patient_data[
                "patient_id"
            ],
            patient_data[
                outcome_column
            ]
        )
    )


    return outcomes


# =============================================================================
# CREATE LONGITUDINAL DATASET
# =============================================================================

def create_longitudinal_dataset(
    wide_longitudinal_data
):
    """
    Convert wide visit-level longitudinal data into LongGCN format.

    Parameters
    ----------
    wide_longitudinal_data : pandas.DataFrame
        One row per patient observation time with measurement columns x1-x5.

    Returns
    -------
    longitudinal_dataset : LongitudinalDataset
        Generalized LongGCN longitudinal representation.

    observation_level_data : pandas.DataFrame
        Observation-level representation used to create the dataset.
    """

    # -------------------------------------------------------------------------
    # 1. CONVERT TO OBSERVATION-LEVEL FORMAT
    # -------------------------------------------------------------------------

    observation_level_data = (
        convert_to_longgcn_format(

            longitudinal_data=wide_longitudinal_data,

            measurement_columns=MEASUREMENT_COLUMNS
        )
    )


    # -------------------------------------------------------------------------
    # 2. CREATE LONGGCN LONGITUDINAL DATASET
    # -------------------------------------------------------------------------

    longitudinal_dataset = LongitudinalDataset(

        data=observation_level_data,

        patient_col="patient_id",

        time_col="time",

        measurement_col="measurement",

        value_col="value"
    )


    return (
        longitudinal_dataset,
        observation_level_data
    )


# =============================================================================
# CREATE DESIGNED MEASUREMENT GROUPS
# =============================================================================

def create_measurement_groups(
    group_definition
):
    """
    Create the designed measurement-group specification.

    Parameters
    ----------
    group_definition : dict
        Mapping from group name to measurement names.

    Returns
    -------
    measurement_groups : DesignedMeasurementGroups
        LongGCN measurement-group specification.
    """

    measurement_groups = DesignedMeasurementGroups(

        measurements=MEASUREMENT_COLUMNS,

        groups=group_definition
    )


    return measurement_groups


# =============================================================================
# CREATE LONGGCN PYTORCH DATASET
# =============================================================================

def create_torch_dataset(
    longitudinal_dataset,
    measurement_groups,
    patient_data,
    outcome_column
):
    """
    Construct a model-ready LongGCNTorchDataset.

    Parameters
    ----------
    longitudinal_dataset : LongitudinalDataset
        Generalized patient-level longitudinal representation.

    measurement_groups : DesignedMeasurementGroups
        Measurement-group specification.

    patient_data : pandas.DataFrame
        Patient-level data for the corresponding split.

    outcome_column : str
        Outcome used for this prediction task.

    Returns
    -------
    torch_dataset : LongGCNTorchDataset
        Precomputed model-ready patient dataset.
    """

    # -------------------------------------------------------------------------
    # 1. CREATE OUTCOME LOOKUP
    # -------------------------------------------------------------------------

    outcomes = create_outcome_mapping(

        patient_data=patient_data,

        outcome_column=outcome_column
    )


    # -------------------------------------------------------------------------
    # 2. CREATE MODEL-READY LONGGCN DATASET
    # -------------------------------------------------------------------------

    torch_dataset = LongGCNTorchDataset(

        longitudinal_dataset=longitudinal_dataset,

        measurement_groups=measurement_groups,

        decay_parameter=DECAY_PARAMETER,

        outcomes=outcomes,

        precompute=PRECOMPUTE
    )


    return torch_dataset


# =============================================================================
# 10. INITIALIZE SUMMARY STORAGE
# =============================================================================

dataset_summaries = []


# =============================================================================
# 11. PREPARE EACH DATA CONDITION
# =============================================================================

for condition_name, condition_settings in DATA_CONDITIONS.items():

    print()
    print("=" * 80)
    print(
        f"DATA CONDITION: {condition_name}"
    )
    print("=" * 80)


    # -------------------------------------------------------------------------
    # CREATE CONDITION OUTPUT DIRECTORY
    # -------------------------------------------------------------------------

    condition_directory = (
        OUTPUT_DIRECTORY
        /
        condition_name
    )


    condition_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    # -------------------------------------------------------------------------
    # CREATE DESIGNED MEASUREMENT GROUPS
    # -------------------------------------------------------------------------

    measurement_groups = (
        create_measurement_groups(
            condition_settings[
                "groups"
            ]
        )
    )


    # -------------------------------------------------------------------------
    # SAVE GROUP DEFINITION
    # -------------------------------------------------------------------------

    group_definition_path = (
        condition_directory
        /
        "measurement_groups.json"
    )


    with open(
        group_definition_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            condition_settings[
                "groups"
            ],
            file,
            indent=4
        )


    # =========================================================================
    # 12. PREPARE EACH TRAIN / VALIDATION / TEST SPLIT
    # =========================================================================

    for split_name in DATA_SPLITS:

        print()
        print(
            f"Preparing split: {split_name}"
        )


        # ---------------------------------------------------------------------
        # LOAD LONGITUDINAL DATA
        # ---------------------------------------------------------------------

        longitudinal_path = (
            SPLIT_DIRECTORY
            /
            (
                f"{split_name}_"
                f"{condition_settings['file_label']}_"
                "longitudinal_data.csv"
            )
        )


        wide_longitudinal_data = pd.read_csv(
            longitudinal_path
        )


        # ---------------------------------------------------------------------
        # LOAD PATIENT DATA
        # ---------------------------------------------------------------------

        patient_path = (
            SPLIT_DIRECTORY
            /
            f"{split_name}_patient_data.csv"
        )


        patient_data = pd.read_csv(
            patient_path
        )


        # ---------------------------------------------------------------------
        # CREATE LONGGCN LONGITUDINAL REPRESENTATION
        # ---------------------------------------------------------------------

        (
            longitudinal_dataset,
            observation_level_data

        ) = create_longitudinal_dataset(
            wide_longitudinal_data
        )


        print(
            "  Patients:",
            len(
                longitudinal_dataset.patient_ids
            )
        )


        print(
            "  Observation-level rows:",
            len(
                observation_level_data
            )
        )


        # =====================================================================
        # 13. CREATE DATASET FOR EACH PREDICTION TASK
        # =====================================================================

        for task_name, outcome_column in PREDICTION_TASKS.items():

            print(
                f"  Creating {task_name} dataset..."
            )


            # -----------------------------------------------------------------
            # CREATE TASK OUTPUT DIRECTORY
            # -----------------------------------------------------------------

            task_directory = (
                condition_directory
                /
                task_name
            )


            task_directory.mkdir(
                parents=True,
                exist_ok=True
            )


            # -----------------------------------------------------------------
            # CREATE MODEL-READY PYTORCH DATASET
            # -----------------------------------------------------------------

            torch_dataset = create_torch_dataset(

                longitudinal_dataset=longitudinal_dataset,

                measurement_groups=measurement_groups,

                patient_data=patient_data,

                outcome_column=outcome_column
            )


            # -----------------------------------------------------------------
            # SAVE LONGGCN DATASET
            # -----------------------------------------------------------------

            dataset_path = (
                task_directory
                /
                f"{split_name}_dataset.pt"
            )


            torch.save(
                torch_dataset,
                dataset_path
            )


            # -----------------------------------------------------------------
            # GET FIRST PATIENT FOR SUMMARY
            # -----------------------------------------------------------------

            example_sample = (
                torch_dataset[
                    0
                ]
            )


            # -----------------------------------------------------------------
            # SAVE DATASET SUMMARY
            # -----------------------------------------------------------------

            dataset_summaries.append(
                {
                    "condition":
                        condition_name,

                    "task":
                        task_name,

                    "split":
                        split_name,

                    "number_of_patients":
                        len(
                            torch_dataset
                        ),

                    "number_of_measurements":
                        len(
                            MEASUREMENT_COLUMNS
                        ),

                    "number_of_groups":
                        len(
                            measurement_groups.group_names
                        ),

                    "group_names":
                        ", ".join(
                            measurement_groups.group_names
                        ),

                    "decay_parameter":
                        DECAY_PARAMETER,

                    "outcome_column":
                        outcome_column,

                    "precomputed":
                        PRECOMPUTE,

                    "example_number_of_times":
                        example_sample.X.shape[
                            0
                        ]
                }
            )


            print(
                "    Saved:",
                dataset_path
            )


# =============================================================================
# 14. SAVE DATASET SUMMARY
# =============================================================================

dataset_summary = pd.DataFrame(
    dataset_summaries
)


summary_path = (
    OUTPUT_DIRECTORY
    /
    "dataset_summary.csv"
)


dataset_summary.to_csv(
    summary_path,
    index=False
)


# =============================================================================
# 15. PRINT FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("LONGGCN DATA PREPARATION COMPLETE")
print("=" * 80)

print()
print(
    dataset_summary.to_string(
        index=False
    )
)

print()
print(
    "Dataset summary saved to:"
)

print(
    summary_path
)

print()