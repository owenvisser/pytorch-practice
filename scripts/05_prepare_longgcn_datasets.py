"""
Prepare LongGCN PyTorch Datasets
================================

This script converts the finalized longitudinal-data conditions into
model-ready LongGCN PyTorch datasets.

It should be run after:

    01_generate_simulated_data.py
    02_generate_incomplete_datasets.py
    03_create_data_splits.py
    04_impute_with_mice3d.py

Eight modeling conditions are prepared:

    1. complete

       Original fully observed data.

    2. mcar

       MCAR missingness is left unfilled.

    3. mcar_to_complete

       The same MCAR data after MICE3D imputation to a complete structure.

    4. group_specific

       Original group-specific observation structure.

    5. group_specific_to_complete

       Group-specific data after MICE3D fills the structurally absent
       measurements.

    6. group_specific_mcar

       Group-specific structure with additional MCAR missingness left
       unfilled.

    7. group_specific_mcar_to_complete

       Group-specific + MCAR data after MICE3D fills all missing
       measurements.

    8. group_specific_mcar_to_group_specific

       Group-specific + MCAR data after MICE3D repairs only the additional
       MCAR missingness. Structural group-specific missingness remains.

Conditions whose final representation is complete or MCAR-only use one
measurement group containing all five measurements.

Conditions whose final representation retains the group-specific structure
use:

    group_1:
        x1, x2, x3

    group_2:
        x4, x5

For every condition, LongGCN PyTorch datasets are created separately for:

    classification
    regression

and for:

    train
    validation
    test

The resulting LongGCNTorchDataset objects contain precomputed patient-specific:

    X
    T
    P
    A
    y

representations.

Run from the PyTorch-Practice project root using:

    python scripts/05_prepare_longgcn_datasets.py
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
# 2. INPUT DIRECTORIES
# =============================================================================

SPLIT_DIRECTORY = (
    PROJECT_ROOT
    /
    "data"
    /
    "splits"
)


IMPUTED_DIRECTORY = (
    PROJECT_ROOT
    /
    "data"
    /
    "imputed"
)


# =============================================================================
# 3. OUTPUT DIRECTORY
# =============================================================================

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
# 4. MEASUREMENT SETTINGS
# =============================================================================

MEASUREMENT_COLUMNS = [

    "x1",

    "x2",

    "x3",

    "x4",

    "x5"
]


# =============================================================================
# 5. TEMPORAL WEIGHTING SETTINGS
# =============================================================================

# Temporal weights are constructed as:
#
#     exp(-delta_t / d)
#
# Because T and A are precomputed inside LongGCNTorchDataset, changing this
# parameter requires rerunning this script.

DECAY_PARAMETER = 0.75


# =============================================================================
# 6. PRECOMPUTATION
# =============================================================================

PRECOMPUTE = True


# =============================================================================
# 7. MEASUREMENT-GROUP DEFINITIONS
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
# 8. MODELING CONDITIONS
# =============================================================================

# source_type:
#
#     "split"
#         Dataset produced directly by scripts/03_create_data_splits.py
#
#     "imputed"
#         Dataset produced by scripts/04_impute_with_mice3d.py
#
#
# final_structure:
#
#     "complete"
#         All measurements are treated as belonging to one measurement group.
#
#     "group_specific"
#         The predefined two-group structure remains in the model input.


DATA_CONDITIONS = {

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    "complete": {

        "source_type":
            "split",

        "file_label":
            "complete",

        "final_structure":
            "complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    # -------------------------------------------------------------------------
    # MCAR OBSERVED-ONLY
    # -------------------------------------------------------------------------

    "mcar": {

        "source_type":
            "split",

        "file_label":
            "mcar",

        "final_structure":
            "complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    # -------------------------------------------------------------------------
    # MCAR -> COMPLETE WITH MICE3D
    # -------------------------------------------------------------------------

    "mcar_to_complete": {

        "source_type":
            "imputed",

        "file_label":
            "mcar_to_complete",

        "final_structure":
            "complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    # -------------------------------------------------------------------------
    # GROUP-SPECIFIC OBSERVED-ONLY
    # -------------------------------------------------------------------------

    "group_specific": {

        "source_type":
            "split",

        "file_label":
            "group_specific",

        "final_structure":
            "group_specific",

        "groups":
            TWO_GROUP_DEFINITION
    },


    # -------------------------------------------------------------------------
    # GROUP-SPECIFIC -> COMPLETE WITH MICE3D
    # -------------------------------------------------------------------------

    "group_specific_to_complete": {

        "source_type":
            "imputed",

        "file_label":
            "group_specific_to_complete",

        "final_structure":
            "complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    # -------------------------------------------------------------------------
    # GROUP-SPECIFIC + MCAR OBSERVED-ONLY
    # -------------------------------------------------------------------------

    "group_specific_mcar": {

        "source_type":
            "split",

        "file_label":
            "group_specific_mcar",

        "final_structure":
            "group_specific",

        "groups":
            TWO_GROUP_DEFINITION
    },


    # -------------------------------------------------------------------------
    # GROUP-SPECIFIC + MCAR -> COMPLETE WITH MICE3D
    # -------------------------------------------------------------------------

    "group_specific_mcar_to_complete": {

        "source_type":
            "imputed",

        "file_label":
            "group_specific_mcar_to_complete",

        "final_structure":
            "complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    # -------------------------------------------------------------------------
    # GROUP-SPECIFIC + MCAR -> GROUP-SPECIFIC WITH MICE3D
    # -------------------------------------------------------------------------

    "group_specific_mcar_to_group_specific": {

        "source_type":
            "imputed",

        "file_label":
            "group_specific_mcar_to_group_specific",

        "final_structure":
            "group_specific",

        "groups":
            TWO_GROUP_DEFINITION
    }
}


# =============================================================================
# 9. PREDICTION TASKS
# =============================================================================

PREDICTION_TASKS = {

    "classification":
        "classification_outcome",

    "regression":
        "regression_outcome"
}


# =============================================================================
# 10. DATA SPLITS
# =============================================================================

DATA_SPLITS = [

    "train",

    "validation",

    "test"
]


# =============================================================================
# GET LONGITUDINAL DATA PATH
# =============================================================================

def get_longitudinal_data_path(
    split_name,
    condition_settings
):
    """
    Construct the longitudinal-data path for one condition and split.
    """

    source_type = (
        condition_settings[
            "source_type"
        ]
    )


    file_label = (
        condition_settings[
            "file_label"
        ]
    )


    if source_type == "split":

        return (

            SPLIT_DIRECTORY

            /

            (
                f"{split_name}_"
                f"{file_label}_"
                "longitudinal_data.csv"
            )
        )


    if source_type == "imputed":

        return (

            IMPUTED_DIRECTORY

            /

            (
                f"{split_name}_"
                f"{file_label}_"
                "mice3d_longitudinal_data.csv"
            )
        )


    raise ValueError(
        "source_type must be either 'split' or 'imputed'."
    )


# =============================================================================
# CREATE OUTCOME MAPPING
# =============================================================================

def create_outcome_mapping(
    patient_data,
    outcome_column
):
    """
    Create a mapping from patient ID to patient-level outcome.
    """

    if outcome_column not in patient_data.columns:

        raise ValueError(
            f"Outcome column '{outcome_column}' "
            "was not found in patient_data."
        )


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
    Convert wide visit-level data into LongGCN observation-level format.
    """

    # -------------------------------------------------------------------------
    # 1. CONVERT WIDE DATA TO OBSERVATION-LEVEL DATA
    # -------------------------------------------------------------------------

    observation_level_data = (
        convert_to_longgcn_format(

            longitudinal_data=(
                wide_longitudinal_data
            ),

            measurement_columns=(
                MEASUREMENT_COLUMNS
            )
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
# CREATE MEASUREMENT GROUPS
# =============================================================================

def create_measurement_groups(
    group_definition
):
    """
    Create the LongGCN measurement-group specification.
    """

    measurement_groups = DesignedMeasurementGroups(

        measurements=MEASUREMENT_COLUMNS,

        groups=group_definition
    )


    return measurement_groups


# =============================================================================
# CREATE PYTORCH DATASET
# =============================================================================

def create_torch_dataset(
    longitudinal_dataset,
    measurement_groups,
    patient_data,
    outcome_column
):
    """
    Create one precomputed LongGCNTorchDataset.
    """

    outcomes = create_outcome_mapping(

        patient_data=patient_data,

        outcome_column=outcome_column
    )


    torch_dataset = LongGCNTorchDataset(

        longitudinal_dataset=(
            longitudinal_dataset
        ),

        measurement_groups=(
            measurement_groups
        ),

        decay_parameter=(
            DECAY_PARAMETER
        ),

        outcomes=outcomes,

        outcome_dtype=torch.float32,

        precompute=PRECOMPUTE
    )


    return torch_dataset


# =============================================================================
# VALIDATE PATIENT SET
# =============================================================================

def validate_patient_set(
    longitudinal_dataset,
    patient_data,
    condition_name,
    split_name
):
    """
    Confirm that the longitudinal and patient-level files contain the same
    patients.
    """

    longitudinal_patient_ids = set(
        longitudinal_dataset.patient_ids
    )


    outcome_patient_ids = set(
        patient_data[
            "patient_id"
        ]
        .tolist()
    )


    if longitudinal_patient_ids != outcome_patient_ids:

        missing_from_longitudinal = (

            outcome_patient_ids

            -

            longitudinal_patient_ids
        )


        missing_from_patient_data = (

            longitudinal_patient_ids

            -

            outcome_patient_ids
        )


        raise ValueError(

            f"{condition_name}, {split_name}: "
            "patient IDs do not match between longitudinal "
            "and patient-level data.\n"
            f"Missing from longitudinal: "
            f"{sorted(missing_from_longitudinal)}\n"
            f"Missing from patient data: "
            f"{sorted(missing_from_patient_data)}"
        )


# =============================================================================
# VALIDATE FINAL DATA STRUCTURE
# =============================================================================

def validate_final_structure(
    wide_longitudinal_data,
    final_structure,
    condition_name,
    split_name
):
    """
    Perform basic checks on the finalized modeling condition.
    """

    number_missing = int(

        wide_longitudinal_data[
            MEASUREMENT_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )


    # Complete-like MICE3D conditions should contain no remaining missing
    # measurements.

    complete_imputation_conditions = {

        "mcar_to_complete",

        "group_specific_to_complete",

        "group_specific_mcar_to_complete"
    }


    if (
        condition_name
        in
        complete_imputation_conditions
        and
        number_missing != 0
    ):

        raise ValueError(

            f"{condition_name}, {split_name}: "
            "this condition should be complete after MICE3D, "
            f"but {number_missing} measurements remain missing."
        )


    # A group-specific target should retain structural missingness.

    if (
        condition_name
        ==
        "group_specific_mcar_to_group_specific"
        and
        number_missing == 0
    ):

        raise ValueError(

            f"{condition_name}, {split_name}: "
            "the target should retain group-specific structural "
            "missingness, but no missing values were found."
        )


    return number_missing


# =============================================================================
# 11. VERIFY REQUIRED INPUT FILES EXIST
# =============================================================================

for condition_name, condition_settings in DATA_CONDITIONS.items():

    for split_name in DATA_SPLITS:

        longitudinal_path = (
            get_longitudinal_data_path(

                split_name=split_name,

                condition_settings=(
                    condition_settings
                )
            )
        )


        if not longitudinal_path.exists():

            raise FileNotFoundError(

                f"Required longitudinal dataset was not found:\n"
                f"{longitudinal_path}"
            )


for split_name in DATA_SPLITS:

    patient_path = (

        SPLIT_DIRECTORY

        /

        f"{split_name}_patient_data.csv"
    )


    if not patient_path.exists():

        raise FileNotFoundError(

            f"Required patient-level dataset was not found:\n"
            f"{patient_path}"
        )


# =============================================================================
# 12. INITIALIZE SUMMARY STORAGE
# =============================================================================

dataset_summaries = []


# =============================================================================
# 13. PREPARE EACH MODELING CONDITION
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "PREPARING LONGGCN DATASETS"
)

print(
    "=" * 80
)


for condition_name, condition_settings in DATA_CONDITIONS.items():

    print()

    print(
        "=" * 80
    )

    print(
        f"CONDITION: {condition_name}"
    )

    print(
        "Source:",
        condition_settings[
            "source_type"
        ]
    )

    print(
        "Final structure:",
        condition_settings[
            "final_structure"
        ]
    )

    print(
        "=" * 80
    )


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
    # CREATE MEASUREMENT GROUPS
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
    # 14. PREPARE TRAIN / VALIDATION / TEST
    # =========================================================================

    for split_name in DATA_SPLITS:

        print()

        print(
            f"Preparing {split_name}..."
        )


        # ---------------------------------------------------------------------
        # LOAD LONGITUDINAL DATA
        # ---------------------------------------------------------------------

        longitudinal_path = (
            get_longitudinal_data_path(

                split_name=split_name,

                condition_settings=(
                    condition_settings
                )
            )
        )


        wide_longitudinal_data = pd.read_csv(
            longitudinal_path
        )


        # ---------------------------------------------------------------------
        # VALIDATE FINAL STRUCTURE
        # ---------------------------------------------------------------------

        number_missing_measurements = (
            validate_final_structure(

                wide_longitudinal_data=(
                    wide_longitudinal_data
                ),

                final_structure=(
                    condition_settings[
                        "final_structure"
                    ]
                ),

                condition_name=condition_name,

                split_name=split_name
            )
        )


        total_measurement_cells = (

            len(
                wide_longitudinal_data
            )

            *

            len(
                MEASUREMENT_COLUMNS
            )
        )


        missing_rate = (

            number_missing_measurements

            /

            total_measurement_cells
        )


        # ---------------------------------------------------------------------
        # LOAD PATIENT-LEVEL DATA
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


        # ---------------------------------------------------------------------
        # VALIDATE PATIENT IDS
        # ---------------------------------------------------------------------

        validate_patient_set(

            longitudinal_dataset=(
                longitudinal_dataset
            ),

            patient_data=patient_data,

            condition_name=condition_name,

            split_name=split_name
        )


        # ---------------------------------------------------------------------
        # DISPLAY DATASET INFORMATION
        # ---------------------------------------------------------------------

        print(
            "  Patients:",
            len(
                longitudinal_dataset.patient_ids
            )
        )


        print(
            "  Visits:",
            len(
                wide_longitudinal_data
            )
        )


        print(
            "  Observed scalar measurements:",
            len(
                observation_level_data
            )
        )


        print(
            "  Missing rate:",
            round(
                missing_rate,
                4
            )
        )


        print(
            "  Measurement groups:",
            list(
                measurement_groups.group_names
            )
        )


        # =====================================================================
        # 15. PREPARE CLASSIFICATION AND REGRESSION
        # =====================================================================

        for task_name, outcome_column in PREDICTION_TASKS.items():

            print(
                f"    Creating {task_name} dataset..."
            )


            # -----------------------------------------------------------------
            # CREATE TASK DIRECTORY
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
            # CREATE PYTORCH DATASET
            # -----------------------------------------------------------------

            torch_dataset = (
                create_torch_dataset(

                    longitudinal_dataset=(
                        longitudinal_dataset
                    ),

                    measurement_groups=(
                        measurement_groups
                    ),

                    patient_data=(
                        patient_data
                    ),

                    outcome_column=(
                        outcome_column
                    )
                )
            )


            # -----------------------------------------------------------------
            # SAVE DATASET
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
            # GET EXAMPLE PATIENT
            # -----------------------------------------------------------------

            example_sample = (
                torch_dataset[
                    0
                ]
            )


            # -----------------------------------------------------------------
            # SAVE SUMMARY
            # -----------------------------------------------------------------

            dataset_summaries.append(
                {
                    "condition":
                        condition_name,

                    "source_type":
                        condition_settings[
                            "source_type"
                        ],

                    "final_structure":
                        condition_settings[
                            "final_structure"
                        ],

                    "task":
                        task_name,

                    "split":
                        split_name,

                    "number_of_patients":
                        len(
                            torch_dataset
                        ),

                    "number_of_visits":
                        len(
                            wide_longitudinal_data
                        ),

                    "observed_measurements":
                        len(
                            observation_level_data
                        ),

                    "missing_measurements":
                        number_missing_measurements,

                    "missing_rate":
                        missing_rate,

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
                        ],

                    "example_X_shape":
                        str(
                            tuple(
                                example_sample.X.shape
                            )
                        ),

                    "example_T_shape":
                        str(
                            tuple(
                                example_sample.T.shape
                            )
                        ),

                    "example_P_shape":
                        str(
                            tuple(
                                example_sample.P.shape
                            )
                        ),

                    "example_A_shape":
                        str(
                            tuple(
                                example_sample.A.shape
                            )
                        )
                }
            )


            print(
                "      Saved:",
                dataset_path
            )


# =============================================================================
# 16. SAVE DATASET SUMMARY
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
# 17. FINAL SUMMARY
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "LONGGCN DATA PREPARATION COMPLETE"
)

print(
    "=" * 80
)


print()

print(
    "Number of modeling conditions:",
    len(
        DATA_CONDITIONS
    )
)


print(
    "Number of prediction tasks:",
    len(
        PREDICTION_TASKS
    )
)


print(
    "Number of data splits:",
    len(
        DATA_SPLITS
    )
)


print(
    "Total saved LongGCN datasets:",
    len(
        dataset_summaries
    )
)


print()

print(
    "Expected total:"
)

print(
    f"  {len(DATA_CONDITIONS)} conditions"
    f" x {len(PREDICTION_TASKS)} tasks"
    f" x {len(DATA_SPLITS)} splits"
    f" = "
    f"{len(DATA_CONDITIONS) * len(PREDICTION_TASKS) * len(DATA_SPLITS)}"
)


print()

print(
    "Dataset summary saved to:"
)

print(
    summary_path
)


print()