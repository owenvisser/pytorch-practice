"""
Create MICE3D-Imputed Comparison Datasets
=========================================

This script creates the MICE3D-derived datasets used in the LongGCN simulation
comparison.

The original data conditions created before this script are NEVER overwritten.

Original conditions retained for modeling:

    1. complete

    2. mcar

    3. group_specific

    4. group_specific_mcar


Four additional MICE3D conditions are created:

    5. mcar_to_complete

       Starting data:
           MCAR

       Target structure:
           complete

       MICE3D fills the measurements removed by MCAR.


    6. group_specific_to_complete

       Starting data:
           group-specific

       Target structure:
           complete

       MICE3D fills the structurally unobserved measurements.


    7. group_specific_mcar_to_complete

       Starting data:
           group-specific + MCAR

       Target structure:
           complete

       MICE3D fills both structural missingness and additional MCAR
       missingness.


    8. group_specific_mcar_to_group_specific

       Starting data:
           group-specific + MCAR

       Target structure:
           group-specific

       MICE3D is run normally, but structural missingness is restored
       afterward using the corresponding group-specific dataset.

       Consequently, only the additional MCAR missingness remains imputed.


Train, validation, and test are each imputed independently using the installed
MICE3D implementation.

Run from the project root using:

    python scripts/04_impute_with_mice3d.py
"""


from pathlib import Path
import sys

import pandas as pd


# =============================================================================
# 1. PROJECT PATH
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        )
    )


from src.mice3d_imputation import (
    impute_with_mice3d,
    restore_target_missingness_structure,
    evaluate_imputation_against_target
)


# =============================================================================
# 2. DIRECTORIES
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


IMPUTED_DIRECTORY.mkdir(
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
# 4. MICE3D SETTINGS
# =============================================================================

MICE_MAX_ITER = 10

GP_ALPHA = 0.001

MICE3D_RANDOM_STATE = 500


# =============================================================================
# 5. IMPUTATION CONDITIONS
# =============================================================================

IMPUTATION_CONDITIONS = {

    "mcar_to_complete": {

        "starting_condition":
            "mcar",

        "target_condition":
            "complete"
    },


    "group_specific_to_complete": {

        "starting_condition":
            "group_specific",

        "target_condition":
            "complete"
    },


    "group_specific_mcar_to_complete": {

        "starting_condition":
            "group_specific_mcar",

        "target_condition":
            "complete"
    },


    "group_specific_mcar_to_group_specific": {

        "starting_condition":
            "group_specific_mcar",

        "target_condition":
            "group_specific"
    }
}


# =============================================================================
# 6. DATA SPLITS
# =============================================================================

DATA_SPLITS = [

    "train",

    "validation",

    "test"
]


# =============================================================================
# 7. INITIALIZE RESULTS
# =============================================================================

imputation_summaries = []

feature_metric_summaries = []


# =============================================================================
# 8. RUN IMPUTATION CONDITIONS
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "MICE3D COMPARISON DATASETS"
)

print(
    "=" * 80
)


for condition_name, condition_settings in IMPUTATION_CONDITIONS.items():

    starting_condition = (
        condition_settings[
            "starting_condition"
        ]
    )


    target_condition = (
        condition_settings[
            "target_condition"
        ]
    )


    print()

    print(
        "=" * 80
    )

    print(
        f"CONDITION: {condition_name}"
    )

    print(
        f"Starting structure: {starting_condition}"
    )

    print(
        f"Target structure:   {target_condition}"
    )

    print(
        "=" * 80
    )


    # =========================================================================
    # 9. TRAIN / VALIDATION / TEST
    # =========================================================================

    for split_name in DATA_SPLITS:

        print()

        print(
            f"Imputing {split_name}..."
        )


        # ---------------------------------------------------------------------
        # LOAD STARTING DATA
        # ---------------------------------------------------------------------

        starting_path = (

            SPLIT_DIRECTORY

            /

            (
                f"{split_name}_"
                f"{starting_condition}_"
                "longitudinal_data.csv"
            )
        )


        starting_data = pd.read_csv(
            starting_path
        )


        # ---------------------------------------------------------------------
        # LOAD TARGET STRUCTURE
        # ---------------------------------------------------------------------

        target_path = (

            SPLIT_DIRECTORY

            /

            (
                f"{split_name}_"
                f"{target_condition}_"
                "longitudinal_data.csv"
            )
        )


        target_data = pd.read_csv(
            target_path
        )


        # ---------------------------------------------------------------------
        # STARTING MISSINGNESS
        # ---------------------------------------------------------------------

        total_cells = (

            len(
                starting_data
            )

            *

            len(
                MEASUREMENT_COLUMNS
            )
        )


        missing_starting = int(

            starting_data[
                MEASUREMENT_COLUMNS
            ]
            .isna()
            .sum()
            .sum()
        )


        starting_missing_rate = (

            missing_starting

            /

            total_cells
        )


        # ---------------------------------------------------------------------
        # TARGET MISSINGNESS
        # ---------------------------------------------------------------------

        missing_target = int(

            target_data[
                MEASUREMENT_COLUMNS
            ]
            .isna()
            .sum()
            .sum()
        )


        target_missing_rate = (

            missing_target

            /

            total_cells
        )


        print(
            "  Patients:",
            starting_data[
                "patient_id"
            ]
            .nunique()
        )


        print(
            "  Starting missing rate:",
            round(
                starting_missing_rate,
                4
            )
        )


        print(
            "  Target missing rate:",
            round(
                target_missing_rate,
                4
            )
        )


        # ---------------------------------------------------------------------
        # RUN MICE3D
        # ---------------------------------------------------------------------

        (
            fully_imputed_data,
            diagnostics

        ) = impute_with_mice3d(

            longitudinal_data=starting_data,

            measurement_columns=MEASUREMENT_COLUMNS,

            mice_max_iter=MICE_MAX_ITER,

            gp_alpha=GP_ALPHA,

            random_state=MICE3D_RANDOM_STATE
        )


        # ---------------------------------------------------------------------
        # RESTORE TARGET MISSINGNESS STRUCTURE
        # ---------------------------------------------------------------------

        # For complete targets, this changes nothing.
        #
        # For:
        #
        #     group_specific_mcar -> group_specific
        #
        # the structural group-specific missing values are restored to NaN.

        final_imputed_data = (
            restore_target_missingness_structure(

                imputed_longitudinal_data=(
                    fully_imputed_data
                ),

                target_longitudinal_data=(
                    target_data
                ),

                measurement_columns=(
                    MEASUREMENT_COLUMNS
                )
            )
        )


        # ---------------------------------------------------------------------
        # FINAL MISSINGNESS
        # ---------------------------------------------------------------------

        missing_final = int(

            final_imputed_data[
                MEASUREMENT_COLUMNS
            ]
            .isna()
            .sum()
            .sum()
        )


        final_missing_rate = (

            missing_final

            /

            total_cells
        )


        # ---------------------------------------------------------------------
        # VERIFY FINAL MISSINGNESS MATCHES TARGET STRUCTURE
        # ---------------------------------------------------------------------

        final_missing_mask = (

            final_imputed_data[
                MEASUREMENT_COLUMNS
            ]
            .isna()
            .to_numpy()
        )


        target_missing_mask = (

            target_data[
                MEASUREMENT_COLUMNS
            ]
            .isna()
            .to_numpy()
        )


        if not (
            final_missing_mask
            ==
            target_missing_mask
        ).all():

            raise RuntimeError(
                f"{condition_name}, {split_name}: "
                "final missingness does not match target structure."
            )


        # ---------------------------------------------------------------------
        # EVALUATE MICE3D AGAINST TARGET
        # ---------------------------------------------------------------------

        (
            overall_metrics,
            feature_metrics

        ) = evaluate_imputation_against_target(

            starting_longitudinal_data=(
                starting_data
            ),

            imputed_longitudinal_data=(
                final_imputed_data
            ),

            target_longitudinal_data=(
                target_data
            ),

            measurement_columns=(
                MEASUREMENT_COLUMNS
            )
        )


        # ---------------------------------------------------------------------
        # SAVE IMPUTED DATASET
        # ---------------------------------------------------------------------

        output_path = (

            IMPUTED_DIRECTORY

            /

            (
                f"{split_name}_"
                f"{condition_name}_"
                "mice3d_longitudinal_data.csv"
            )
        )


        final_imputed_data.to_csv(

            output_path,

            index=False
        )


        # ---------------------------------------------------------------------
        # SAVE SUMMARY
        # ---------------------------------------------------------------------

        imputation_summaries.append(
            {
                "condition":
                    condition_name,

                "split":
                    split_name,

                "starting_condition":
                    starting_condition,

                "target_condition":
                    target_condition,

                "number_of_patients":
                    diagnostics[
                        "number_of_patients"
                    ],

                "maximum_number_of_visits":
                    diagnostics[
                        "maximum_number_of_visits"
                    ],

                "total_measurement_cells":
                    total_cells,

                "missing_starting":
                    missing_starting,

                "starting_missing_rate":
                    starting_missing_rate,

                "missing_target":
                    missing_target,

                "target_missing_rate":
                    target_missing_rate,

                "missing_final":
                    missing_final,

                "final_missing_rate":
                    final_missing_rate,

                "number_imputed":
                    overall_metrics[
                        "number_imputed"
                    ],

                "imputation_mae":
                    overall_metrics[
                        "mae"
                    ],

                "imputation_rmse":
                    overall_metrics[
                        "rmse"
                    ],

                "mice_max_iter":
                    MICE_MAX_ITER,

                "gp_alpha":
                    GP_ALPHA,

                "random_state":
                    MICE3D_RANDOM_STATE
            }
        )


        # ---------------------------------------------------------------------
        # FEATURE-SPECIFIC METRICS
        # ---------------------------------------------------------------------

        feature_metrics = (
            feature_metrics.copy()
        )


        feature_metrics.insert(
            0,
            "target_condition",
            target_condition
        )


        feature_metrics.insert(
            0,
            "starting_condition",
            starting_condition
        )


        feature_metrics.insert(
            0,
            "split",
            split_name
        )


        feature_metrics.insert(
            0,
            "condition",
            condition_name
        )


        feature_metric_summaries.append(
            feature_metrics
        )


        # ---------------------------------------------------------------------
        # PRINT RESULTS
        # ---------------------------------------------------------------------

        print(
            "  Final missing rate:",
            round(
                final_missing_rate,
                4
            )
        )


        print(
            "  Values actually imputed toward target:",
            overall_metrics[
                "number_imputed"
            ]
        )


        print(
            "  Imputation MAE:",
            round(
                overall_metrics[
                    "mae"
                ],
                6
            )
        )


        print(
            "  Imputation RMSE:",
            round(
                overall_metrics[
                    "rmse"
                ],
                6
            )
        )


        print(
            "  Saved:",
            output_path
        )


# =============================================================================
# 10. SAVE OVERALL SUMMARY
# =============================================================================

imputation_summary = pd.DataFrame(
    imputation_summaries
)


summary_path = (

    IMPUTED_DIRECTORY

    /

    "mice3d_imputation_summary.csv"
)


imputation_summary.to_csv(

    summary_path,

    index=False
)


# =============================================================================
# 11. SAVE FEATURE-SPECIFIC SUMMARY
# =============================================================================

feature_metrics_summary = pd.concat(

    feature_metric_summaries,

    ignore_index=True
)


feature_summary_path = (

    IMPUTED_DIRECTORY

    /

    "mice3d_feature_metrics.csv"
)


feature_metrics_summary.to_csv(

    feature_summary_path,

    index=False
)


# =============================================================================
# 12. FINAL SUMMARY
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "MICE3D IMPUTATION COMPLETE"
)

print(
    "=" * 80
)


print()

print(
    imputation_summary.to_string(
        index=False
    )
)


print()

print(
    "Summary saved to:"
)

print(
    summary_path
)


print()

print(
    "Feature metrics saved to:"
)

print(
    feature_summary_path
)


print()