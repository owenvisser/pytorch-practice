"""
Prepare LongGCN Temporal-Decay Variants
=======================================

This script extends the baseline LongGCN data-preparation experiment by
constructing multiple graph representations using different values of the
temporal-decay parameter d.

The current LongGCN temporal weighting is:

    w(delta_t; d) = exp(-delta_t / d)

where:

    delta_t
        is the elapsed time between two observations

    d
        controls how quickly temporal influence decays

Smaller d:
    stronger temporal decay
    only nearby observations contribute substantial weight

Larger d:
    slower temporal decay
    more distant observations retain greater influence


This script uses the exact same:

    longitudinal datasets
    train / validation / test splits
    measurement-group definitions
    prediction outcomes

as scripts/05_prepare_longgcn_datasets.py.

Only the temporal-decay parameter changes.

The baseline datasets created by script 05 are NOT modified.

Variant datasets are saved under:

    data/longgcn_variants/exponential/


Later, alternative temporal weighting functions can be incorporated after
the corresponding functionality has been added to the longgcn package.

Run from the PyTorch-Practice project root using:

    python scripts/05b_prepare_longgcn_variants.py
"""


from pathlib import Path
import json
import sys

import pandas as pd
import torch


# =============================================================================
# 1. PROJECT PATH
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
    "longgcn_variants"
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 4. MEASUREMENTS
# =============================================================================

MEASUREMENT_COLUMNS = [

    "x1",
    "x2",
    "x3",
    "x4",
    "x5"
]


# =============================================================================
# 5. TEMPORAL WEIGHTING
# =============================================================================

# The current validated longgcn temporal weighting is:
#
#     exp(-delta_t / d)
#
# The weighting-function label is explicitly retained in the output structure
# so additional kernels can be incorporated later without changing the overall
# experiment organization.

TEMPORAL_WEIGHTING = "exponential"


DECAY_PARAMETERS = [

    0.25,

    0.50,

    0.75,

    1.00,

    2.00,

    5.00
]


# =============================================================================
# 6. PRECOMPUTATION
# =============================================================================

PRECOMPUTE = True


# =============================================================================
# 7. GROUP DEFINITIONS
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

DATA_CONDITIONS = {

    "complete": {

        "source_type":
            "split",

        "file_label":
            "complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    "mcar": {

        "source_type":
            "split",

        "file_label":
            "mcar",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    "mcar_to_complete": {

        "source_type":
            "imputed",

        "file_label":
            "mcar_to_complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    "group_specific": {

        "source_type":
            "split",

        "file_label":
            "group_specific",

        "groups":
            TWO_GROUP_DEFINITION
    },


    "group_specific_to_complete": {

        "source_type":
            "imputed",

        "file_label":
            "group_specific_to_complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    "group_specific_mcar": {

        "source_type":
            "split",

        "file_label":
            "group_specific_mcar",

        "groups":
            TWO_GROUP_DEFINITION
    },


    "group_specific_mcar_to_complete": {

        "source_type":
            "imputed",

        "file_label":
            "group_specific_mcar_to_complete",

        "groups":
            SINGLE_GROUP_DEFINITION
    },


    "group_specific_mcar_to_group_specific": {

        "source_type":
            "imputed",

        "file_label":
            "group_specific_mcar_to_group_specific",

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
# FORMAT DECAY PARAMETER FOR FILE PATH
# =============================================================================

def format_decay_parameter(
    decay_parameter
):
    """
    Convert a numerical d value into a filesystem-friendly label.

    Examples
    --------

    0.25 -> d_0p25
    0.75 -> d_0p75
    1.00 -> d_1
    5.00 -> d_5
    """

    text = f"{decay_parameter:g}"

    text = text.replace(
        ".",
        "p"
    )


    return f"d_{text}"


# =============================================================================
# GET LONGITUDINAL DATA PATH
# =============================================================================

def get_longitudinal_data_path(
    split_name,
    condition_settings
):
    """
    Return the finalized longitudinal-data path for one condition and split.
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
    Map patient IDs to patient-level prediction outcomes.
    """

    return dict(

        zip(

            patient_data[
                "patient_id"
            ],

            patient_data[
                outcome_column
            ]
        )
    )


# =============================================================================
# CREATE LONGITUDINAL DATASET
# =============================================================================

def create_longitudinal_dataset(
    wide_longitudinal_data
):
    """
    Convert wide simulated longitudinal data into LongGCN observation format.
    """

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


    longitudinal_dataset = LongitudinalDataset(

        data=observation_level_data,

        patient_col="patient_id",

        time_col="time",

        measurement_col="measurement",

        value_col="value"
    )


    return longitudinal_dataset


# =============================================================================
# CREATE MEASUREMENT GROUPS
# =============================================================================

def create_measurement_groups(
    group_definition
):
    """
    Create the designed measurement-group object.
    """

    return DesignedMeasurementGroups(

        measurements=MEASUREMENT_COLUMNS,

        groups=group_definition
    )


# =============================================================================
# CREATE LONGGCN TORCH DATASET
# =============================================================================

def create_torch_dataset(
    longitudinal_dataset,
    measurement_groups,
    patient_data,
    outcome_column,
    decay_parameter
):
    """
    Create one LongGCNTorchDataset using the requested temporal d value.
    """

    outcomes = create_outcome_mapping(

        patient_data=patient_data,

        outcome_column=outcome_column
    )


    return LongGCNTorchDataset(

        longitudinal_dataset=(
            longitudinal_dataset
        ),

        measurement_groups=(
            measurement_groups
        ),

        decay_parameter=(
            decay_parameter
        ),

        outcomes=outcomes,

        outcome_dtype=torch.float32,

        precompute=PRECOMPUTE
    )


# =============================================================================
# 11. VERIFY INPUT FILES
# =============================================================================

for condition_settings in DATA_CONDITIONS.values():

    for split_name in DATA_SPLITS:

        input_path = get_longitudinal_data_path(

            split_name=split_name,

            condition_settings=(
                condition_settings
            )
        )


        if not input_path.exists():

            raise FileNotFoundError(
                f"Required dataset was not found:\n{input_path}"
            )


# =============================================================================
# 12. PRELOAD RAW LONGITUDINAL DATA
# =============================================================================

# The underlying longitudinal observations do not depend on d.
#
# We therefore construct each LongitudinalDataset only once and reuse it while
# building the different graph variants.

prepared_data = {}


print()

print(
    "=" * 80
)

print(
    "LOADING LONGITUDINAL DATA"
)

print(
    "=" * 80
)


for condition_name, condition_settings in DATA_CONDITIONS.items():

    prepared_data[
        condition_name
    ] = {}


    measurement_groups = (
        create_measurement_groups(

            condition_settings[
                "groups"
            ]
        )
    )


    for split_name in DATA_SPLITS:

        longitudinal_path = (
            get_longitudinal_data_path(

                split_name=split_name,

                condition_settings=(
                    condition_settings
                )
            )
        )


        patient_path = (

            SPLIT_DIRECTORY

            /

            f"{split_name}_patient_data.csv"
        )


        wide_longitudinal_data = pd.read_csv(
            longitudinal_path
        )


        patient_data = pd.read_csv(
            patient_path
        )


        longitudinal_dataset = (
            create_longitudinal_dataset(

                wide_longitudinal_data
            )
        )


        prepared_data[
            condition_name
        ][
            split_name
        ] = {

            "longitudinal_dataset":
                longitudinal_dataset,

            "patient_data":
                patient_data,

            "measurement_groups":
                measurement_groups
        }


# =============================================================================
# 13. CREATE TEMPORAL-DECAY VARIANTS
# =============================================================================

summary_rows = []


total_variants = (

    len(
        DECAY_PARAMETERS
    )

    *

    len(
        DATA_CONDITIONS
    )

    *

    len(
        PREDICTION_TASKS
    )

    *

    len(
        DATA_SPLITS
    )
)


variant_number = 0


print()

print(
    "=" * 80
)

print(
    "PREPARING LONGGCN TEMPORAL VARIANTS"
)

print(
    "=" * 80
)


print()

print(
    "Temporal weighting:",
    TEMPORAL_WEIGHTING
)


print(
    "d values:",
    DECAY_PARAMETERS
)


print(
    "Total datasets to create:",
    total_variants
)


# =============================================================================
# 14. LOOP OVER d
# =============================================================================

for decay_parameter in DECAY_PARAMETERS:

    decay_label = (
        format_decay_parameter(
            decay_parameter
        )
    )


    decay_directory = (

        OUTPUT_DIRECTORY

        /

        TEMPORAL_WEIGHTING

        /

        decay_label
    )


    decay_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    print()

    print(
        "#" * 80
    )

    print(
        f"d = {decay_parameter}"
    )

    print(
        "#" * 80
    )


    # =========================================================================
    # 15. LOOP OVER DATA CONDITIONS
    # =========================================================================

    for condition_name, condition_settings in DATA_CONDITIONS.items():

        condition_directory = (

            decay_directory

            /

            condition_name
        )


        condition_directory.mkdir(
            parents=True,
            exist_ok=True
        )


        # ---------------------------------------------------------------------
        # SAVE GROUP DEFINITION
        # ---------------------------------------------------------------------

        group_path = (

            condition_directory

            /

            "measurement_groups.json"
        )


        with open(
            group_path,
            "w",
            encoding="utf-8"
        ) as group_file:

            json.dump(

                condition_settings[
                    "groups"
                ],

                group_file,

                indent=4
            )


        # =====================================================================
        # 16. LOOP OVER SPLITS
        # =====================================================================

        for split_name in DATA_SPLITS:

            split_data = (

                prepared_data[
                    condition_name
                ][
                    split_name
                ]
            )


            # =================================================================
            # 17. LOOP OVER PREDICTION TASKS
            # =================================================================

            for (
                task_name,
                outcome_column
            ) in PREDICTION_TASKS.items():

                variant_number += 1


                print(

                    f"[{variant_number:3d}/{total_variants}] "

                    f"d={decay_parameter:g} | "
                    f"{condition_name} | "
                    f"{task_name} | "
                    f"{split_name}"
                )


                task_directory = (

                    condition_directory

                    /

                    task_name
                )


                task_directory.mkdir(
                    parents=True,
                    exist_ok=True
                )


                torch_dataset = (
                    create_torch_dataset(

                        longitudinal_dataset=(
                            split_data[
                                "longitudinal_dataset"
                            ]
                        ),

                        measurement_groups=(
                            split_data[
                                "measurement_groups"
                            ]
                        ),

                        patient_data=(
                            split_data[
                                "patient_data"
                            ]
                        ),

                        outcome_column=(
                            outcome_column
                        ),

                        decay_parameter=(
                            decay_parameter
                        )
                    )
                )


                dataset_path = (

                    task_directory

                    /

                    f"{split_name}_dataset.pt"
                )


                torch.save(

                    torch_dataset,

                    dataset_path
                )


                summary_rows.append({

                    "temporal_weighting":
                        TEMPORAL_WEIGHTING,

                    "decay_parameter":
                        decay_parameter,

                    "decay_label":
                        decay_label,

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

                    "number_of_groups":
                        len(
                            split_data[
                                "measurement_groups"
                            ]
                            .group_names
                        ),

                    "dataset_path":
                        str(
                            dataset_path
                            .relative_to(
                                PROJECT_ROOT
                            )
                        )
                })


# =============================================================================
# 18. SAVE VARIANT INDEX
# =============================================================================

variant_summary = pd.DataFrame(
    summary_rows
)


variant_summary_path = (

    OUTPUT_DIRECTORY

    /

    "variant_dataset_index.csv"
)


variant_summary.to_csv(

    variant_summary_path,

    index=False
)


# =============================================================================
# 19. SAVE EXPERIMENT GRID
# =============================================================================

experiment_grid = pd.DataFrame({

    "temporal_weighting":
        [
            TEMPORAL_WEIGHTING
        ]
        *
        len(
            DECAY_PARAMETERS
        ),

    "decay_parameter":
        DECAY_PARAMETERS,

    "decay_label":
        [
            format_decay_parameter(
                value
            )
            for value
            in DECAY_PARAMETERS
        ]
})


experiment_grid.to_csv(

    OUTPUT_DIRECTORY

    /

    "temporal_variant_grid.csv",

    index=False
)


# =============================================================================
# 20. FINAL SUMMARY
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "TEMPORAL VARIANT PREPARATION COMPLETE"
)

print(
    "=" * 80
)


print()

print(
    "Temporal weighting:",
    TEMPORAL_WEIGHTING
)


print(
    "Number of d values:",
    len(
        DECAY_PARAMETERS
    )
)


print(
    "Data conditions:",
    len(
        DATA_CONDITIONS
    )
)


print(
    "Prediction tasks:",
    len(
        PREDICTION_TASKS
    )
)


print(
    "Data splits:",
    len(
        DATA_SPLITS
    )
)


print(
    "Total saved datasets:",
    len(
        variant_summary
    )
)


print()

print(
    "Variant index:"
)

print(
    variant_summary_path
)


print()