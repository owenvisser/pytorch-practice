"""
Summarize LongGCN Model Results
===============================

This script summarizes the model-fitting results produced by:

    scripts/06_run_longgcn_models.py

The primary goal is to compare predictive performance across the eight
missing-data and imputation conditions.

The script produces:

    1. Classification balanced-accuracy plot

    2. Classification sensitivity/specificity plot

    3. Regression RMSE plot

    4. Regression R-squared plot

    5. Classification imputation-gain plot

    6. Regression imputation-gain plot

    7. Combined model comparison table

    8. Pairwise imputation comparison table


Interpretation of the imputation-gain plots
-------------------------------------------

Classification:

    gain =
        balanced accuracy after imputation
        -
        balanced accuracy without imputation

Regression:

    gain =
        RMSE without imputation
        -
        RMSE after imputation

Therefore, for BOTH plots:

    positive values
        imputation improved prediction

    zero
        essentially no predictive difference

    negative values
        imputation reduced predictive performance


Run from the PyTorch-Practice project root using:

    python scripts/07_summarize_model_results.py
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


RESULTS_DIRECTORY = (

    PROJECT_ROOT
    /
    "results"
)


FIGURE_DIRECTORY = (

    RESULTS_DIRECTORY
    /
    "figures"
)


FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_PATH = (

    RESULTS_DIRECTORY
    /
    "longgcn_model_results.csv"
)


# =============================================================================
# 2. CONDITION ORDER
# =============================================================================

# This order reflects the scientific comparisons rather than alphabetical
# ordering.

CONDITION_ORDER = [

    "complete",

    "mcar",

    "mcar_to_complete",

    "group_specific",

    "group_specific_to_complete",

    "group_specific_mcar",

    "group_specific_mcar_to_group_specific",

    "group_specific_mcar_to_complete"
]


# =============================================================================
# 3. DISPLAY LABELS
# =============================================================================

CONDITION_LABELS = {

    "complete":
        "Complete",

    "mcar":
        "MCAR",

    "mcar_to_complete":
        "MCAR → Complete",

    "group_specific":
        "Group-specific",

    "group_specific_to_complete":
        "Group-specific → Complete",

    "group_specific_mcar":
        "Group-specific + MCAR",

    "group_specific_mcar_to_group_specific":
        "Group + MCAR → Group-specific",

    "group_specific_mcar_to_complete":
        "Group + MCAR → Complete"
}


# =============================================================================
# 4. LOAD RESULTS
# =============================================================================

if not RESULTS_PATH.exists():

    raise FileNotFoundError(

        "Combined model results were not found:\n"
        f"{RESULTS_PATH}"
    )


results = pd.read_csv(
    RESULTS_PATH
)


# =============================================================================
# 5. SPLIT CLASSIFICATION AND REGRESSION RESULTS
# =============================================================================

classification_results = (

    results[
        results["task"]
        ==
        "classification"
    ]
    .copy()
)


regression_results = (

    results[
        results["task"]
        ==
        "regression"
    ]
    .copy()
)


# =============================================================================
# 6. APPLY CONDITION ORDER
# =============================================================================

for data in [
    classification_results,
    regression_results
]:

    data["condition"] = pd.Categorical(

        data["condition"],

        categories=CONDITION_ORDER,

        ordered=True
    )


    data.sort_values(
        "condition",
        inplace=True
    )


    data["condition_label"] = (

        data["condition"]
        .astype(str)
        .map(
            CONDITION_LABELS
        )
    )


# =============================================================================
# HELPER: ADD VALUE LABELS TO HORIZONTAL BARS
# =============================================================================

def label_horizontal_bars(
    axis,
    number_format=".3f"
):
    """
    Place numerical values at the end of horizontal bars.
    """

    for bar in axis.patches:

        width = bar.get_width()


        axis.text(

            width,

            bar.get_y()
            +
            bar.get_height() / 2,

            f" {width:{number_format}}",

            va="center"
        )


# =============================================================================
# 7. CLASSIFICATION: BALANCED ACCURACY
# =============================================================================

figure, axis = plt.subplots(
    figsize=(
        10,
        6
    )
)


axis.barh(

    classification_results[
        "condition_label"
    ],

    classification_results[
        "balanced_accuracy"
    ]
)


axis.set_xlim(
    0,
    1
)


axis.set_xlabel(
    "Balanced Accuracy"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Classification Performance"
)


axis.invert_yaxis()


label_horizontal_bars(
    axis
)


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY
    /
    "classification_balanced_accuracy.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 8. CLASSIFICATION: SENSITIVITY AND SPECIFICITY
# =============================================================================

figure, axis = plt.subplots(
    figsize=(
        11,
        6
    )
)


positions = np.arange(
    len(
        classification_results
    )
)


bar_height = 0.35


axis.barh(

    positions
    -
    bar_height / 2,

    classification_results[
        "sensitivity"
    ],

    height=bar_height,

    label="Sensitivity"
)


axis.barh(

    positions
    +
    bar_height / 2,

    classification_results[
        "specificity"
    ],

    height=bar_height,

    label="Specificity"
)


axis.set_yticks(
    positions
)


axis.set_yticklabels(

    classification_results[
        "condition_label"
    ]
)


axis.set_xlim(
    0,
    1
)


axis.set_xlabel(
    "Classification Rate"
)


axis.set_title(
    "Classification Sensitivity and Specificity"
)


axis.legend()


axis.invert_yaxis()


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY
    /
    "classification_sensitivity_specificity.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 9. REGRESSION: RMSE
# =============================================================================

figure, axis = plt.subplots(
    figsize=(
        10,
        6
    )
)


axis.barh(

    regression_results[
        "condition_label"
    ],

    regression_results[
        "rmse"
    ]
)


axis.set_xlabel(
    "RMSE"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Regression Performance"
)


axis.invert_yaxis()


label_horizontal_bars(
    axis
)


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY
    /
    "regression_rmse.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 10. REGRESSION: R-SQUARED
# =============================================================================

figure, axis = plt.subplots(
    figsize=(
        10,
        6
    )
)


axis.barh(

    regression_results[
        "condition_label"
    ],

    regression_results[
        "r2"
    ]
)


axis.set_xlabel(
    "R²"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Regression R²"
)


axis.invert_yaxis()


label_horizontal_bars(
    axis
)


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY
    /
    "regression_r2.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 11. DEFINE IMPUTATION COMPARISONS
# =============================================================================

IMPUTATION_COMPARISONS = {

    "MCAR → Complete": {

        "observed":
            "mcar",

        "imputed":
            "mcar_to_complete"
    },


    "Group-specific → Complete": {

        "observed":
            "group_specific",

        "imputed":
            "group_specific_to_complete"
    },


    "Group + MCAR → Group-specific": {

        "observed":
            "group_specific_mcar",

        "imputed":
            "group_specific_mcar_to_group_specific"
    },


    "Group + MCAR → Complete": {

        "observed":
            "group_specific_mcar",

        "imputed":
            "group_specific_mcar_to_complete"
    }
}


# =============================================================================
# 12. CREATE IMPUTATION COMPARISON TABLE
# =============================================================================

imputation_rows = []


for (
    comparison_name,
    comparison
) in IMPUTATION_COMPARISONS.items():

    observed_condition = (
        comparison[
            "observed"
        ]
    )


    imputed_condition = (
        comparison[
            "imputed"
        ]
    )


    # -------------------------------------------------------------------------
    # CLASSIFICATION
    # -------------------------------------------------------------------------

    observed_classification = (

        classification_results[
            classification_results[
                "condition"
            ]
            ==
            observed_condition
        ]
        .iloc[0]
    )


    imputed_classification = (

        classification_results[
            classification_results[
                "condition"
            ]
            ==
            imputed_condition
        ]
        .iloc[0]
    )


    classification_gain = (

        imputed_classification[
            "balanced_accuracy"
        ]

        -

        observed_classification[
            "balanced_accuracy"
        ]
    )


    # -------------------------------------------------------------------------
    # REGRESSION
    # -------------------------------------------------------------------------

    observed_regression = (

        regression_results[
            regression_results[
                "condition"
            ]
            ==
            observed_condition
        ]
        .iloc[0]
    )


    imputed_regression = (

        regression_results[
            regression_results[
                "condition"
            ]
            ==
            imputed_condition
        ]
        .iloc[0]
    )


    # Positive means lower RMSE after imputation.

    regression_gain = (

        observed_regression[
            "rmse"
        ]

        -

        imputed_regression[
            "rmse"
        ]
    )


    imputation_rows.append({

        "comparison":
            comparison_name,

        "observed_condition":
            observed_condition,

        "imputed_condition":
            imputed_condition,

        "observed_balanced_accuracy":
            observed_classification[
                "balanced_accuracy"
            ],

        "imputed_balanced_accuracy":
            imputed_classification[
                "balanced_accuracy"
            ],

        "balanced_accuracy_gain":
            classification_gain,

        "observed_rmse":
            observed_regression[
                "rmse"
            ],

        "imputed_rmse":
            imputed_regression[
                "rmse"
            ],

        "rmse_gain":
            regression_gain
    })


imputation_summary = pd.DataFrame(
    imputation_rows
)


imputation_summary.to_csv(

    RESULTS_DIRECTORY
    /
    "imputation_comparison_summary.csv",

    index=False
)


# =============================================================================
# 13. CLASSIFICATION IMPUTATION GAIN
# =============================================================================

figure, axis = plt.subplots(
    figsize=(
        10,
        5
    )
)


axis.barh(

    imputation_summary[
        "comparison"
    ],

    imputation_summary[
        "balanced_accuracy_gain"
    ]
)


axis.axvline(
    x=0,
    linewidth=1
)


axis.set_xlabel(
    "Change in Balanced Accuracy"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Classification Gain from MICE3D Imputation"
)


axis.invert_yaxis()


for bar in axis.patches:

    value = bar.get_width()


    axis.text(

        value,

        bar.get_y()
        +
        bar.get_height() / 2,

        f" {value:+.3f}",

        va="center"
    )


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY
    /
    "classification_imputation_gain.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 14. REGRESSION IMPUTATION GAIN
# =============================================================================

figure, axis = plt.subplots(
    figsize=(
        10,
        5
    )
)


axis.barh(

    imputation_summary[
        "comparison"
    ],

    imputation_summary[
        "rmse_gain"
    ]
)


axis.axvline(
    x=0,
    linewidth=1
)


axis.set_xlabel(
    "Reduction in RMSE"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Regression Gain from MICE3D Imputation"
)


axis.invert_yaxis()


for bar in axis.patches:

    value = bar.get_width()


    axis.text(

        value,

        bar.get_y()
        +
        bar.get_height() / 2,

        f" {value:+.3f}",

        va="center"
    )


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY
    /
    "regression_imputation_gain.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 15. CREATE COMBINED MODEL SUMMARY TABLE
# =============================================================================

classification_table = classification_results[

    [
        "condition",
        "number_of_groups",
        "total_parameters",
        "best_epoch",
        "balanced_accuracy",
        "sensitivity",
        "specificity"
    ]
].copy()


classification_table = (
    classification_table.rename(
        columns={

            "best_epoch":
                "classification_best_epoch",

            "balanced_accuracy":
                "balanced_accuracy",

            "sensitivity":
                "sensitivity",

            "specificity":
                "specificity"
        }
    )
)


regression_table = regression_results[

    [
        "condition",
        "best_epoch",
        "rmse",
        "mae",
        "r2"
    ]
].copy()


regression_table = (
    regression_table.rename(
        columns={

            "best_epoch":
                "regression_best_epoch"
        }
    )
)


comparison_table = classification_table.merge(

    regression_table,

    on="condition",

    how="inner"
)


comparison_table["condition_label"] = (

    comparison_table[
        "condition"
    ]
    .astype(str)
    .map(
        CONDITION_LABELS
    )
)


comparison_table = comparison_table[

    [
        "condition",
        "condition_label",
        "number_of_groups",
        "total_parameters",
        "classification_best_epoch",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "regression_best_epoch",
        "rmse",
        "mae",
        "r2"
    ]
]


comparison_table.to_csv(

    RESULTS_DIRECTORY
    /
    "model_comparison_summary.csv",

    index=False
)


# =============================================================================
# 16. CREATE PRESENTATION-READY TABLE IMAGE
# =============================================================================

display_table = comparison_table[

    [
        "condition_label",
        "total_parameters",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "rmse",
        "mae",
        "r2"
    ]
].copy()


display_table = display_table.rename(
    columns={

        "condition_label":
            "Condition",

        "total_parameters":
            "Parameters",

        "balanced_accuracy":
            "Balanced Acc.",

        "sensitivity":
            "Sensitivity",

        "specificity":
            "Specificity",

        "rmse":
            "RMSE",

        "mae":
            "MAE",

        "r2":
            "R²"
    }
)


for column in [

    "Balanced Acc.",

    "Sensitivity",

    "Specificity",

    "RMSE",

    "MAE",

    "R²"
]:

    display_table[column] = (

        display_table[column]
        .map(
            lambda value:
            f"{value:.3f}"
        )
    )


figure, axis = plt.subplots(
    figsize=(
        15,
        5.5
    )
)


axis.axis(
    "off"
)


table = axis.table(

    cellText=(
        display_table.values
    ),

    colLabels=(
        display_table.columns
    ),

    cellLoc="center",

    loc="center"
)


table.auto_set_font_size(
    False
)


table.set_fontsize(
    10
)


table.scale(
    1,
    1.7
)


axis.set_title(

    "LongGCN Prediction Performance Across Missing-Data Conditions",

    pad=20
)


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY
    /
    "model_comparison_table.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 17. PRINT RESULTS
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "MODEL RESULTS SUMMARY"
)

print(
    "=" * 80
)


print()

print(
    "CLASSIFICATION"
)

print(
    "--------------"
)


print(

    classification_results[

        [
            "condition_label",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "total_parameters",
            "best_epoch"
        ]
    ]
    .to_string(
        index=False
    )
)


print()

print(
    "REGRESSION"
)

print(
    "----------"
)


print(

    regression_results[

        [
            "condition_label",
            "rmse",
            "mae",
            "r2",
            "total_parameters",
            "best_epoch"
        ]
    ]
    .to_string(
        index=False
    )
)


print()

print(
    "IMPUTATION EFFECTS"
)

print(
    "------------------"
)


print(

    imputation_summary[

        [
            "comparison",
            "balanced_accuracy_gain",
            "rmse_gain"
        ]
    ]
    .to_string(
        index=False
    )
)


print()

print(
    "Positive imputation-gain values indicate improved predictive performance."
)


print()

print(
    "Figures saved to:"
)

print(
    FIGURE_DIRECTORY
)


print()