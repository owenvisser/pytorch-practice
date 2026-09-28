"""
Summarize Missing-Data Strategy Results
=======================================

This script summarizes the primary scientific comparison for the LongGCN
simulation study.

The central comparison is NOT:

    LongGCN on completed data
        versus
    Deep ReLU on the same completed data

because LongGCN is intended primarily for settings in which the longitudinal
measurement record is incomplete or structurally heterogeneous.

Instead, the primary comparison is:

    incomplete longitudinal data
        |
        |-----------------------------|
        |                             |
        v                             v
    Direct LongGCN                  MICE3D
                                      |
                                      v
                                  Deep ReLU


The three primary starting-data scenarios are:

    1. MCAR

       Direct strategy:
           MCAR -> LongGCN

       Conventional strategy:
           MCAR -> MICE3D -> complete data -> Deep ReLU


    2. Group-specific observation

       Direct strategy:
           Group-specific -> LongGCN

       Conventional strategy:
           Group-specific -> MICE3D -> complete data -> Deep ReLU


    3. Group-specific observation + MCAR

       Direct strategy:
           Group-specific + MCAR -> LongGCN

       Conventional strategy:
           Group-specific + MCAR -> MICE3D -> complete data -> Deep ReLU

       An additional selective-repair strategy is also retained:

           Group-specific + MCAR
               -> MICE3D
               -> restore structural group missingness
               -> LongGCN


Why retain the selective-repair strategy?
-----------------------------------------

The selective-repair strategy asks a scientifically different question:

    Can accidental missingness be repaired while preserving the designed
    measurement-group structure that motivated LongGCN?

This produces three strategies for the Group-specific + MCAR condition:

    1. Direct LongGCN

    2. MICE3D repairs accidental MCAR only
       -> restore structural missingness
       -> LongGCN

    3. MICE3D fills everything
       -> Deep ReLU


Primary metrics
---------------

Classification:
    balanced accuracy
    sensitivity
    specificity

Regression:
    RMSE
    MAE
    R^2


Comparison-direction convention
-------------------------------

For classification:

    conventional_balanced_accuracy_gain
        =
        BA(MICE3D + Deep ReLU)
        -
        BA(Direct LongGCN)

Positive values favor MICE3D + Deep ReLU.


For regression:

    conventional_rmse_gain
        =
        RMSE(Direct LongGCN)
        -
        RMSE(MICE3D + Deep ReLU)

Positive values favor MICE3D + Deep ReLU because lower RMSE is better.


Validation-loss curves
----------------------

The script also creates full validation-loss curves using every saved epoch.

Different line endpoints therefore display early stopping directly.

Within those figures:

    color
        identifies the original missing-data scenario

    solid line
        Direct LongGCN

    dashed line
        MICE3D + Deep ReLU

    dotted line
        selective MICE3D repair + LongGCN
        for Group-specific + MCAR only


Expected input files
--------------------

    results/
        longgcn_model_results.csv
        deep_relu_model_results.csv

        classification/
            <condition>/
                training_history.csv

        regression/
            <condition>/
                training_history.csv

        deep_relu/
            classification/
                <condition>/
                    training_history.csv

            regression/
                <condition>/
                    training_history.csv


Run from the PyTorch-Practice project root using:

    python scripts/07_summarize_model_results.py
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


LONGGCN_RESULTS_PATH = (
    RESULTS_DIRECTORY
    /
    "longgcn_model_results.csv"
)


DEEP_RELU_RESULTS_PATH = (
    RESULTS_DIRECTORY
    /
    "deep_relu_model_results.csv"
)


# =============================================================================
# 2. OUTPUT PATHS
# =============================================================================

STRATEGY_LEVEL_RESULTS_PATH = (
    RESULTS_DIRECTORY
    /
    "strategy_level_results.csv"
)


PRIMARY_COMPARISON_PATH = (
    RESULTS_DIRECTORY
    /
    "primary_strategy_comparison.csv"
)


GROUP_MCAR_COMPARISON_PATH = (
    RESULTS_DIRECTORY
    /
    "group_mcar_three_strategy_comparison.csv"
)


# =============================================================================
# 3. SCIENTIFIC COMPARISON DEFINITIONS
# =============================================================================

PRIMARY_SCENARIOS = [

    {
        "scenario":
            "MCAR",

        "direct_longgcn_condition":
            "mcar",

        "mice3d_deep_relu_condition":
            "mcar_to_complete"
    },

    {
        "scenario":
            "Group-specific",

        "direct_longgcn_condition":
            "group_specific",

        "mice3d_deep_relu_condition":
            "group_specific_to_complete"
    },

    {
        "scenario":
            "Group-specific + MCAR",

        "direct_longgcn_condition":
            "group_specific_mcar",

        "mice3d_deep_relu_condition":
            "group_specific_mcar_to_complete"
    }
]


# This is the special condition in which MICE3D repairs accidental MCAR
# missingness but the original structural group missingness is restored.

SELECTIVE_REPAIR_CONDITION = (
    "group_specific_mcar_to_group_specific"
)


# Labels used throughout tables and figures.

DIRECT_STRATEGY_LABEL = (
    "Direct LongGCN"
)


CONVENTIONAL_STRATEGY_LABEL = (
    "MICE3D + Deep ReLU"
)


SELECTIVE_REPAIR_LABEL = (
    "MICE3D repair MCAR + LongGCN"
)


# =============================================================================
# 4. LOAD MODEL RESULT FILES
# =============================================================================

def load_results(
    path,
    result_name
):
    """
    Load one model-result CSV and verify required columns.

    Parameters
    ----------
    path : pathlib.Path
        Path to the result CSV.

    result_name : str
        Human-readable name used in error messages.

    Returns
    -------
    results : pandas.DataFrame
        Loaded result table.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{result_name} was not found:\n"
            f"{path}"
        )


    results = pd.read_csv(
        path
    )


    required_columns = {

        "condition",
        "task",
        "total_parameters",
        "best_epoch",
        "best_validation_loss",
        "test_loss"
    }


    missing_columns = (

        required_columns

        -

        set(
            results.columns
        )
    )


    if missing_columns:

        raise ValueError(
            f"{result_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


    return results


longgcn_results = load_results(

    path=LONGGCN_RESULTS_PATH,

    result_name="LongGCN results"
)


deep_relu_results = load_results(

    path=DEEP_RELU_RESULTS_PATH,

    result_name="Deep ReLU results"
)


# =============================================================================
# 5. ACCESS ONE MODEL RESULT
# =============================================================================

def get_result_row(
    results,
    condition,
    task
):
    """
    Return exactly one result row for a condition/task combination.
    """

    matching_rows = results[

        (
            results[
                "condition"
            ]
            ==
            condition
        )

        &

        (
            results[
                "task"
            ]
            ==
            task
        )
    ]


    if len(
        matching_rows
    ) != 1:

        raise ValueError(

            "Expected exactly one result row for:\n"

            f"    condition = {condition}\n"

            f"    task      = {task}\n"

            f"but found {len(matching_rows)} rows."
        )


    return matching_rows.iloc[
        0
    ]


# =============================================================================
# 6. BUILD TIDY STRATEGY-LEVEL RESULTS
# =============================================================================

def build_strategy_level_results():
    """
    Create one row per scenario / strategy combination.

    Primary scenarios receive two rows:

        Direct LongGCN
        MICE3D + Deep ReLU

    Group-specific + MCAR receives a third row:

        MICE3D repair MCAR + LongGCN
    """

    rows = []


    # =========================================================================
    # PRIMARY TWO-STRATEGY COMPARISONS
    # =========================================================================

    for scenario_definition in PRIMARY_SCENARIOS:

        scenario = (
            scenario_definition[
                "scenario"
            ]
        )


        direct_condition = (
            scenario_definition[
                "direct_longgcn_condition"
            ]
        )


        conventional_condition = (
            scenario_definition[
                "mice3d_deep_relu_condition"
            ]
        )


        # ---------------------------------------------------------------------
        # DIRECT LONGGCN: CLASSIFICATION
        # ---------------------------------------------------------------------

        direct_classification = get_result_row(

            results=longgcn_results,

            condition=direct_condition,

            task="classification"
        )


        # ---------------------------------------------------------------------
        # DIRECT LONGGCN: REGRESSION
        # ---------------------------------------------------------------------

        direct_regression = get_result_row(

            results=longgcn_results,

            condition=direct_condition,

            task="regression"
        )


        # ---------------------------------------------------------------------
        # SAVE DIRECT LONGGCN ROW
        # ---------------------------------------------------------------------

        rows.append(

            {

                "scenario":
                    scenario,

                "strategy":
                    DIRECT_STRATEGY_LABEL,

                "model":
                    "LongGCN",

                "input_condition":
                    direct_condition,

                "classification_parameters":
                    direct_classification[
                        "total_parameters"
                    ],

                "classification_best_epoch":
                    direct_classification[
                        "best_epoch"
                    ],

                "balanced_accuracy":
                    direct_classification[
                        "balanced_accuracy"
                    ],

                "sensitivity":
                    direct_classification[
                        "sensitivity"
                    ],

                "specificity":
                    direct_classification[
                        "specificity"
                    ],

                "regression_parameters":
                    direct_regression[
                        "total_parameters"
                    ],

                "regression_best_epoch":
                    direct_regression[
                        "best_epoch"
                    ],

                "rmse":
                    direct_regression[
                        "rmse"
                    ],

                "mae":
                    direct_regression[
                        "mae"
                    ],

                "r2":
                    direct_regression[
                        "r2"
                    ]
            }
        )


        # ---------------------------------------------------------------------
        # MICE3D + DEEP RELU: CLASSIFICATION
        # ---------------------------------------------------------------------

        conventional_classification = get_result_row(

            results=deep_relu_results,

            condition=conventional_condition,

            task="classification"
        )


        # ---------------------------------------------------------------------
        # MICE3D + DEEP RELU: REGRESSION
        # ---------------------------------------------------------------------

        conventional_regression = get_result_row(

            results=deep_relu_results,

            condition=conventional_condition,

            task="regression"
        )


        # ---------------------------------------------------------------------
        # SAVE MICE3D + DEEP RELU ROW
        # ---------------------------------------------------------------------

        rows.append(

            {

                "scenario":
                    scenario,

                "strategy":
                    CONVENTIONAL_STRATEGY_LABEL,

                "model":
                    "Deep ReLU",

                "input_condition":
                    conventional_condition,

                "classification_parameters":
                    conventional_classification[
                        "total_parameters"
                    ],

                "classification_best_epoch":
                    conventional_classification[
                        "best_epoch"
                    ],

                "balanced_accuracy":
                    conventional_classification[
                        "balanced_accuracy"
                    ],

                "sensitivity":
                    conventional_classification[
                        "sensitivity"
                    ],

                "specificity":
                    conventional_classification[
                        "specificity"
                    ],

                "regression_parameters":
                    conventional_regression[
                        "total_parameters"
                    ],

                "regression_best_epoch":
                    conventional_regression[
                        "best_epoch"
                    ],

                "rmse":
                    conventional_regression[
                        "rmse"
                    ],

                "mae":
                    conventional_regression[
                        "mae"
                    ],

                "r2":
                    conventional_regression[
                        "r2"
                    ]
            }
        )


    # =========================================================================
    # SELECTIVE REPAIR FOR GROUP-SPECIFIC + MCAR
    # =========================================================================

    selective_classification = get_result_row(

        results=longgcn_results,

        condition=SELECTIVE_REPAIR_CONDITION,

        task="classification"
    )


    selective_regression = get_result_row(

        results=longgcn_results,

        condition=SELECTIVE_REPAIR_CONDITION,

        task="regression"
    )


    rows.append(

        {

            "scenario":
                "Group-specific + MCAR",

            "strategy":
                SELECTIVE_REPAIR_LABEL,

            "model":
                "LongGCN",

            "input_condition":
                SELECTIVE_REPAIR_CONDITION,

            "classification_parameters":
                selective_classification[
                    "total_parameters"
                ],

            "classification_best_epoch":
                selective_classification[
                    "best_epoch"
                ],

            "balanced_accuracy":
                selective_classification[
                    "balanced_accuracy"
                ],

            "sensitivity":
                selective_classification[
                    "sensitivity"
                ],

            "specificity":
                selective_classification[
                    "specificity"
                ],

            "regression_parameters":
                selective_regression[
                    "total_parameters"
                ],

            "regression_best_epoch":
                selective_regression[
                    "best_epoch"
                ],

            "rmse":
                selective_regression[
                    "rmse"
                ],

            "mae":
                selective_regression[
                    "mae"
                ],

            "r2":
                selective_regression[
                    "r2"
                ]
        }
    )


    strategy_results = pd.DataFrame(
        rows
    )


    # -------------------------------------------------------------------------
    # DEFINE DISPLAY ORDER
    # -------------------------------------------------------------------------

    scenario_order = [

        "MCAR",

        "Group-specific",

        "Group-specific + MCAR"
    ]


    strategy_order = [

        DIRECT_STRATEGY_LABEL,

        SELECTIVE_REPAIR_LABEL,

        CONVENTIONAL_STRATEGY_LABEL
    ]


    strategy_results[
        "scenario"
    ] = pd.Categorical(

        strategy_results[
            "scenario"
        ],

        categories=scenario_order,

        ordered=True
    )


    strategy_results[
        "strategy"
    ] = pd.Categorical(

        strategy_results[
            "strategy"
        ],

        categories=strategy_order,

        ordered=True
    )


    strategy_results.sort_values(

        [
            "scenario",
            "strategy"
        ],

        inplace=True
    )


    strategy_results.reset_index(

        drop=True,

        inplace=True
    )


    return strategy_results


strategy_level_results = (
    build_strategy_level_results()
)


strategy_level_results.to_csv(

    STRATEGY_LEVEL_RESULTS_PATH,

    index=False
)


# =============================================================================
# 7. BUILD PRIMARY DIRECT VS CONVENTIONAL COMPARISON
# =============================================================================

def build_primary_comparison(
    strategy_results
):
    """
    Create one comparison row per starting incomplete-data scenario.

    Positive gain values indicate better performance for:

        MICE3D + Deep ReLU.
    """

    rows = []


    for scenario_definition in PRIMARY_SCENARIOS:

        scenario = (
            scenario_definition[
                "scenario"
            ]
        )


        scenario_rows = strategy_results[

            strategy_results[
                "scenario"
            ]
            ==
            scenario
        ]


        direct_row = scenario_rows[

            scenario_rows[
                "strategy"
            ]
            ==
            DIRECT_STRATEGY_LABEL

        ].iloc[
            0
        ]


        conventional_row = scenario_rows[

            scenario_rows[
                "strategy"
            ]
            ==
            CONVENTIONAL_STRATEGY_LABEL

        ].iloc[
            0
        ]


        rows.append(

            {

                "scenario":
                    scenario,

                # =============================================================
                # CLASSIFICATION
                # =============================================================

                "direct_longgcn_balanced_accuracy":
                    direct_row[
                        "balanced_accuracy"
                    ],

                "mice3d_deep_relu_balanced_accuracy":
                    conventional_row[
                        "balanced_accuracy"
                    ],

                "conventional_balanced_accuracy_gain":
                    (
                        conventional_row[
                            "balanced_accuracy"
                        ]
                        -
                        direct_row[
                            "balanced_accuracy"
                        ]
                    ),

                "direct_longgcn_sensitivity":
                    direct_row[
                        "sensitivity"
                    ],

                "mice3d_deep_relu_sensitivity":
                    conventional_row[
                        "sensitivity"
                    ],

                "direct_longgcn_specificity":
                    direct_row[
                        "specificity"
                    ],

                "mice3d_deep_relu_specificity":
                    conventional_row[
                        "specificity"
                    ],

                # =============================================================
                # REGRESSION
                # =============================================================

                "direct_longgcn_rmse":
                    direct_row[
                        "rmse"
                    ],

                "mice3d_deep_relu_rmse":
                    conventional_row[
                        "rmse"
                    ],

                "conventional_rmse_gain":
                    (
                        direct_row[
                            "rmse"
                        ]
                        -
                        conventional_row[
                            "rmse"
                        ]
                    ),

                "direct_longgcn_mae":
                    direct_row[
                        "mae"
                    ],

                "mice3d_deep_relu_mae":
                    conventional_row[
                        "mae"
                    ],

                "direct_longgcn_r2":
                    direct_row[
                        "r2"
                    ],

                "mice3d_deep_relu_r2":
                    conventional_row[
                        "r2"
                    ],

                # =============================================================
                # MODEL SIZE
                # =============================================================

                "direct_classification_parameters":
                    direct_row[
                        "classification_parameters"
                    ],

                "conventional_classification_parameters":
                    conventional_row[
                        "classification_parameters"
                    ]
            }
        )


    return pd.DataFrame(
        rows
    )


primary_comparison = build_primary_comparison(

    strategy_level_results
)


primary_comparison.to_csv(

    PRIMARY_COMPARISON_PATH,

    index=False
)


# =============================================================================
# 8. GROUP + MCAR THREE-STRATEGY COMPARISON
# =============================================================================

group_mcar_comparison = (

    strategy_level_results[

        strategy_level_results[
            "scenario"
        ]
        ==
        "Group-specific + MCAR"
    ]

    .copy()
)


group_mcar_comparison.to_csv(

    GROUP_MCAR_COMPARISON_PATH,

    index=False
)


# =============================================================================
# 9. PLOTTING HELPERS
# =============================================================================

def add_value_labels_horizontal(
    axis,
    bars,
    decimals=3,
    padding_fraction=0.01
):
    """
    Add numeric labels immediately to the right of horizontal bars.
    """

    x_minimum, x_maximum = (
        axis.get_xlim()
    )


    padding = (

        (
            x_maximum
            -
            x_minimum
        )

        *

        padding_fraction
    )


    for bar in bars:

        value = bar.get_width()


        axis.text(

            value
            +
            padding,

            bar.get_y()
            +
            bar.get_height() / 2,

            f"{value:.{decimals}f}",

            va="center",

            ha="left",

            clip_on=False
        )


def save_figure(
    figure,
    filename,
    right_margin=0.82
):
    """
    Apply a consistent layout and save one figure.
    """

    figure.tight_layout(

        rect=[
            0,
            0,
            right_margin,
            1
        ]
    )


    figure.savefig(

        FIGURE_DIRECTORY
        /
        filename,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close(
        figure
    )


# =============================================================================
# 10. PRIMARY CLASSIFICATION COMPARISON
# =============================================================================

scenario_labels = (

    primary_comparison[
        "scenario"
    ]

    .tolist()
)


positions = np.arange(

    len(
        scenario_labels
    )
)


bar_height = 0.34


figure, axis = plt.subplots(

    figsize=(
        11,
        6
    )
)


direct_bars = axis.barh(

    positions
    -
    bar_height / 2,

    primary_comparison[
        "direct_longgcn_balanced_accuracy"
    ],

    height=bar_height,

    label=DIRECT_STRATEGY_LABEL
)


conventional_bars = axis.barh(

    positions
    +
    bar_height / 2,

    primary_comparison[
        "mice3d_deep_relu_balanced_accuracy"
    ],

    height=bar_height,

    label=CONVENTIONAL_STRATEGY_LABEL
)


axis.set_yticks(
    positions
)


axis.set_yticklabels(
    scenario_labels
)


axis.set_xlim(
    0,
    1.05
)


axis.set_xlabel(
    "Balanced Accuracy"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Classification: Direct LongGCN vs MICE3D + Deep ReLU"
)


axis.invert_yaxis()


axis.grid(

    axis="x",

    alpha=0.25
)


axis.legend(

    loc="center left",

    bbox_to_anchor=(
        1.02,
        0.5
    ),

    frameon=False
)


add_value_labels_horizontal(

    axis=axis,

    bars=direct_bars
)


add_value_labels_horizontal(

    axis=axis,

    bars=conventional_bars
)


save_figure(

    figure=figure,

    filename=(
        "classification_primary_strategy_comparison.png"
    ),

    right_margin=0.78
)


# =============================================================================
# 11. PRIMARY REGRESSION COMPARISON
# =============================================================================

figure, axis = plt.subplots(

    figsize=(
        11,
        6
    )
)


direct_bars = axis.barh(

    positions
    -
    bar_height / 2,

    primary_comparison[
        "direct_longgcn_rmse"
    ],

    height=bar_height,

    label=DIRECT_STRATEGY_LABEL
)


conventional_bars = axis.barh(

    positions
    +
    bar_height / 2,

    primary_comparison[
        "mice3d_deep_relu_rmse"
    ],

    height=bar_height,

    label=CONVENTIONAL_STRATEGY_LABEL
)


axis.set_yticks(
    positions
)


axis.set_yticklabels(
    scenario_labels
)


axis.set_xlabel(
    "RMSE"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Regression: Direct LongGCN vs MICE3D + Deep ReLU"
)


axis.invert_yaxis()


axis.grid(

    axis="x",

    alpha=0.25
)


axis.legend(

    loc="center left",

    bbox_to_anchor=(
        1.02,
        0.5
    ),

    frameon=False
)


add_value_labels_horizontal(

    axis=axis,

    bars=direct_bars
)


add_value_labels_horizontal(

    axis=axis,

    bars=conventional_bars
)


save_figure(

    figure=figure,

    filename=(
        "regression_primary_strategy_comparison.png"
    ),

    right_margin=0.78
)


# =============================================================================
# 12. CLASSIFICATION SENSITIVITY
# =============================================================================

figure, axis = plt.subplots(

    figsize=(
        11,
        6
    )
)


direct_bars = axis.barh(

    positions
    -
    bar_height / 2,

    primary_comparison[
        "direct_longgcn_sensitivity"
    ],

    height=bar_height,

    label=DIRECT_STRATEGY_LABEL
)


conventional_bars = axis.barh(

    positions
    +
    bar_height / 2,

    primary_comparison[
        "mice3d_deep_relu_sensitivity"
    ],

    height=bar_height,

    label=CONVENTIONAL_STRATEGY_LABEL
)


axis.set_yticks(
    positions
)


axis.set_yticklabels(
    scenario_labels
)


axis.set_xlim(
    0,
    1.05
)


axis.set_xlabel(
    "Sensitivity"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Classification Sensitivity by Missing-Data Strategy"
)


axis.invert_yaxis()


axis.grid(

    axis="x",

    alpha=0.25
)


axis.legend(

    loc="center left",

    bbox_to_anchor=(
        1.02,
        0.5
    ),

    frameon=False
)


add_value_labels_horizontal(
    axis,
    direct_bars
)


add_value_labels_horizontal(
    axis,
    conventional_bars
)


save_figure(

    figure=figure,

    filename=(
        "classification_primary_strategy_sensitivity.png"
    ),

    right_margin=0.78
)


# =============================================================================
# 13. CLASSIFICATION SPECIFICITY
# =============================================================================

figure, axis = plt.subplots(

    figsize=(
        11,
        6
    )
)


direct_bars = axis.barh(

    positions
    -
    bar_height / 2,

    primary_comparison[
        "direct_longgcn_specificity"
    ],

    height=bar_height,

    label=DIRECT_STRATEGY_LABEL
)


conventional_bars = axis.barh(

    positions
    +
    bar_height / 2,

    primary_comparison[
        "mice3d_deep_relu_specificity"
    ],

    height=bar_height,

    label=CONVENTIONAL_STRATEGY_LABEL
)


axis.set_yticks(
    positions
)


axis.set_yticklabels(
    scenario_labels
)


axis.set_xlim(
    0,
    1.05
)


axis.set_xlabel(
    "Specificity"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Classification Specificity by Missing-Data Strategy"
)


axis.invert_yaxis()


axis.grid(

    axis="x",

    alpha=0.25
)


axis.legend(

    loc="center left",

    bbox_to_anchor=(
        1.02,
        0.5
    ),

    frameon=False
)


add_value_labels_horizontal(
    axis,
    direct_bars
)


add_value_labels_horizontal(
    axis,
    conventional_bars
)


save_figure(

    figure=figure,

    filename=(
        "classification_primary_strategy_specificity.png"
    ),

    right_margin=0.78
)


# =============================================================================
# 14. REGRESSION R-SQUARED
# =============================================================================

figure, axis = plt.subplots(

    figsize=(
        11,
        6
    )
)


direct_bars = axis.barh(

    positions
    -
    bar_height / 2,

    primary_comparison[
        "direct_longgcn_r2"
    ],

    height=bar_height,

    label=DIRECT_STRATEGY_LABEL
)


conventional_bars = axis.barh(

    positions
    +
    bar_height / 2,

    primary_comparison[
        "mice3d_deep_relu_r2"
    ],

    height=bar_height,

    label=CONVENTIONAL_STRATEGY_LABEL
)


axis.set_yticks(
    positions
)


axis.set_yticklabels(
    scenario_labels
)


axis.set_xlabel(
    "R²"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Regression R² by Missing-Data Strategy"
)


axis.invert_yaxis()


axis.grid(

    axis="x",

    alpha=0.25
)


axis.legend(

    loc="center left",

    bbox_to_anchor=(
        1.02,
        0.5
    ),

    frameon=False
)


add_value_labels_horizontal(
    axis,
    direct_bars
)


add_value_labels_horizontal(
    axis,
    conventional_bars
)


save_figure(

    figure=figure,

    filename=(
        "regression_primary_strategy_r2.png"
    ),

    right_margin=0.78
)


# =============================================================================
# 15. GROUP + MCAR THREE-STRATEGY ORDER
# =============================================================================

GROUP_MCAR_STRATEGY_ORDER = [

    DIRECT_STRATEGY_LABEL,

    SELECTIVE_REPAIR_LABEL,

    CONVENTIONAL_STRATEGY_LABEL
]


group_mcar_plot_data = (
    group_mcar_comparison
    .copy()
)


group_mcar_plot_data[
    "strategy"
] = pd.Categorical(

    group_mcar_plot_data[
        "strategy"
    ],

    categories=(
        GROUP_MCAR_STRATEGY_ORDER
    ),

    ordered=True
)


group_mcar_plot_data.sort_values(

    "strategy",

    inplace=True
)


group_mcar_positions = np.arange(

    len(
        group_mcar_plot_data
    )
)


# =============================================================================
# 16. GROUP + MCAR THREE-STRATEGY CLASSIFICATION
# =============================================================================

figure, axis = plt.subplots(

    figsize=(
        11,
        5.5
    )
)


bars = axis.barh(

    group_mcar_positions,

    group_mcar_plot_data[
        "balanced_accuracy"
    ]
)


axis.set_yticks(
    group_mcar_positions
)


axis.set_yticklabels(

    group_mcar_plot_data[
        "strategy"
    ]

    .astype(
        str
    )
)


axis.set_xlim(
    0,
    1.05
)


axis.set_xlabel(
    "Balanced Accuracy"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Group-specific + MCAR: Three Missing-Data Strategies"
)


axis.invert_yaxis()


axis.grid(

    axis="x",

    alpha=0.25
)


add_value_labels_horizontal(

    axis=axis,

    bars=bars
)


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY

    /

    "classification_group_mcar_three_strategy_comparison.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 17. GROUP + MCAR THREE-STRATEGY REGRESSION
# =============================================================================

figure, axis = plt.subplots(

    figsize=(
        11,
        5.5
    )
)


bars = axis.barh(

    group_mcar_positions,

    group_mcar_plot_data[
        "rmse"
    ]
)


axis.set_yticks(
    group_mcar_positions
)


axis.set_yticklabels(

    group_mcar_plot_data[
        "strategy"
    ]

    .astype(
        str
    )
)


axis.set_xlabel(
    "RMSE"
)


axis.set_ylabel(
    ""
)


axis.set_title(
    "Group-specific + MCAR: Three Missing-Data Strategies"
)


axis.invert_yaxis()


axis.grid(

    axis="x",

    alpha=0.25
)


add_value_labels_horizontal(

    axis=axis,

    bars=bars
)


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY

    /

    "regression_group_mcar_three_strategy_comparison.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 18. LOAD EPOCH-LEVEL TRAINING HISTORY
# =============================================================================

def load_training_history(
    model_type,
    task,
    condition
):
    """
    Load one complete epoch-level training history.

    Parameters
    ----------
    model_type : str
        Either:

            "longgcn"
            "deep_relu"

    task : str
        Either:

            "classification"
            "regression"

    condition : str
        Saved experimental condition.
    """

    # -------------------------------------------------------------------------
    # LONGGCN
    # -------------------------------------------------------------------------

    if model_type == "longgcn":

        history_path = (

            RESULTS_DIRECTORY

            /

            task

            /

            condition

            /

            "training_history.csv"
        )


    # -------------------------------------------------------------------------
    # DEEP RELU
    # -------------------------------------------------------------------------

    elif model_type == "deep_relu":

        history_path = (

            RESULTS_DIRECTORY

            /

            "deep_relu"

            /

            task

            /

            condition

            /

            "training_history.csv"
        )


    else:

        raise ValueError(
            "model_type must be 'longgcn' or 'deep_relu'."
        )


    if not history_path.exists():

        raise FileNotFoundError(

            "Training history was not found:\n"

            f"{history_path}"
        )


    history = pd.read_csv(
        history_path
    )


    required_columns = {

        "epoch",

        "validation_loss"
    }


    missing_columns = (

        required_columns

        -

        set(
            history.columns
        )
    )


    if missing_columns:

        raise ValueError(

            f"{history_path} is missing required columns: "

            f"{sorted(missing_columns)}"
        )


    return history


# =============================================================================
# 19. PLOT VALIDATION-LOSS STRATEGY COMPARISON
# =============================================================================

def plot_validation_loss_comparison(
    task,
    output_filename,
    title
):
    """
    Plot complete validation-loss histories for the relevant strategies.

    Color identifies the starting missing-data scenario.

    Solid line:
        Direct LongGCN

    Dashed line:
        MICE3D + Deep ReLU

    Dotted line:
        MICE3D repairs accidental MCAR + LongGCN
        for Group-specific + MCAR only.

    Because every saved epoch is shown, early stopping is visible from the
    endpoint of each line.
    """

    figure, axis = plt.subplots(

        figsize=(
            13,
            7
        )
    )


    color_cycle = (

        plt.rcParams[
            "axes.prop_cycle"
        ]

        .by_key()[

            "color"
        ]
    )


    # =========================================================================
    # PRIMARY DIRECT VS CONVENTIONAL STRATEGIES
    # =========================================================================

    for scenario_index, scenario_definition in enumerate(

        PRIMARY_SCENARIOS
    ):

        scenario = (
            scenario_definition[
                "scenario"
            ]
        )


        color = color_cycle[

            scenario_index

            %

            len(
                color_cycle
            )
        ]


        # ---------------------------------------------------------------------
        # DIRECT LONGGCN
        # ---------------------------------------------------------------------

        direct_history = load_training_history(

            model_type="longgcn",

            task=task,

            condition=(
                scenario_definition[
                    "direct_longgcn_condition"
                ]
            )
        )


        axis.plot(

            direct_history[
                "epoch"
            ],

            direct_history[
                "validation_loss"
            ],

            label=(

                f"{scenario} — "
                f"{DIRECT_STRATEGY_LABEL}"
            ),

            linewidth=1.8,

            linestyle="-",

            color=color
        )


        # ---------------------------------------------------------------------
        # MICE3D + DEEP RELU
        # ---------------------------------------------------------------------

        conventional_history = load_training_history(

            model_type="deep_relu",

            task=task,

            condition=(
                scenario_definition[
                    "mice3d_deep_relu_condition"
                ]
            )
        )


        axis.plot(

            conventional_history[
                "epoch"
            ],

            conventional_history[
                "validation_loss"
            ],

            label=(

                f"{scenario} — "
                f"{CONVENTIONAL_STRATEGY_LABEL}"
            ),

            linewidth=1.8,

            linestyle="--",

            color=color
        )


    # =========================================================================
    # SELECTIVE REPAIR STRATEGY
    # =========================================================================

    selective_history = load_training_history(

        model_type="longgcn",

        task=task,

        condition=SELECTIVE_REPAIR_CONDITION
    )


    selective_color = color_cycle[

        2

        %

        len(
            color_cycle
        )
    ]


    axis.plot(

        selective_history[
            "epoch"
        ],

        selective_history[
            "validation_loss"
        ],

        label=(

            "Group-specific + MCAR — "

            f"{SELECTIVE_REPAIR_LABEL}"
        ),

        linewidth=1.8,

        linestyle=":",

        color=selective_color
    )


    # =========================================================================
    # PLOT FORMATTING
    # =========================================================================

    axis.set_xlabel(
        "Epoch"
    )


    axis.set_ylabel(
        "Validation Loss"
    )


    axis.set_title(
        title
    )


    axis.grid(
        alpha=0.25
    )


    # Legend outside the plotting region on the right.

    axis.legend(

        loc="center left",

        bbox_to_anchor=(
            1.02,
            0.5
        ),

        frameon=False
    )


    save_figure(

        figure=figure,

        filename=output_filename,

        right_margin=0.68
    )


# =============================================================================
# 20. CLASSIFICATION VALIDATION LOSS
# =============================================================================

plot_validation_loss_comparison(

    task="classification",

    output_filename=(
        "classification_validation_loss_strategy_comparison.png"
    ),

    title=(
        "Classification Validation Loss by Missing-Data Strategy"
    )
)


# =============================================================================
# 21. REGRESSION VALIDATION LOSS
# =============================================================================

plot_validation_loss_comparison(

    task="regression",

    output_filename=(
        "regression_validation_loss_strategy_comparison.png"
    ),

    title=(
        "Regression Validation Loss by Missing-Data Strategy"
    )
)


# =============================================================================
# 22. CREATE PRIMARY COMPARISON TABLE IMAGE
# =============================================================================

table_data = strategy_level_results[

    [
        "scenario",
        "strategy",
        "classification_parameters",
        "classification_best_epoch",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "regression_best_epoch",
        "rmse",
        "mae",
        "r2"
    ]

].copy()


table_data = table_data.rename(

    columns={

        "scenario":
            "Scenario",

        "strategy":
            "Strategy",

        "classification_parameters":
            "Params",

        "classification_best_epoch":
            "Class Epoch",

        "balanced_accuracy":
            "BA",

        "sensitivity":
            "Sens",

        "specificity":
            "Spec",

        "regression_best_epoch":
            "Reg Epoch",

        "rmse":
            "RMSE",

        "mae":
            "MAE",

        "r2":
            "R²"
    }
)


# Convert categorical columns to ordinary strings before rendering.

table_data[
    "Scenario"
] = table_data[
    "Scenario"
].astype(
    str
)


table_data[
    "Strategy"
] = table_data[
    "Strategy"
].astype(
    str
)


# -------------------------------------------------------------------------
# FORMAT DECIMAL COLUMNS
# -------------------------------------------------------------------------

for column in [

    "BA",
    "Sens",
    "Spec",
    "RMSE",
    "MAE",
    "R²"

]:

    table_data[
        column
    ] = table_data[
        column
    ].map(

        lambda value:
        f"{value:.3f}"
    )


# -------------------------------------------------------------------------
# FORMAT INTEGER COLUMNS
# -------------------------------------------------------------------------

for column in [

    "Params",
    "Class Epoch",
    "Reg Epoch"

]:

    table_data[
        column
    ] = table_data[
        column
    ].map(

        lambda value:
        f"{int(value)}"
    )


# -------------------------------------------------------------------------
# DRAW TABLE
# -------------------------------------------------------------------------

figure, axis = plt.subplots(

    figsize=(
        18,
        5.8
    )
)


axis.axis(
    "off"
)


table = axis.table(

    cellText=(
        table_data.values
    ),

    colLabels=(
        table_data.columns
    ),

    cellLoc="center",

    loc="center"
)


table.auto_set_font_size(
    False
)


table.set_fontsize(
    9
)


table.scale(
    1,
    1.7
)


axis.set_title(

    "Primary Missing-Data Strategy Comparison",

    pad=20
)


figure.tight_layout()


figure.savefig(

    FIGURE_DIRECTORY

    /

    "primary_strategy_comparison_table.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(
    figure
)


# =============================================================================
# 23. CONSOLE SUMMARY
# =============================================================================

print()

print(
    "=" * 88
)

print(
    "PRIMARY SCIENTIFIC COMPARISON"
)

print(
    "=" * 88
)


print()


print(
    "Incomplete data"
)

print(
    "      |"
)

print(
    "      |-----------------------------|"
)

print(
    "      |                             |"
)

print(
    "      v                             v"
)

print(
    "Direct LongGCN                  MICE3D"
)

print(
    "                                    |"
)

print(
    "                                    v"
)

print(
    "                                Deep ReLU"
)


# =============================================================================
# CLASSIFICATION CONSOLE SUMMARY
# =============================================================================

print()

print(
    "CLASSIFICATION: BALANCED ACCURACY"
)

print(
    "---------------------------------"
)


classification_display = primary_comparison[

    [
        "scenario",
        "direct_longgcn_balanced_accuracy",
        "mice3d_deep_relu_balanced_accuracy",
        "conventional_balanced_accuracy_gain"
    ]

].copy()


print(

    classification_display.to_string(
        index=False
    )
)


print()

print(
    "Positive conventional_balanced_accuracy_gain values favor "
    "MICE3D + Deep ReLU."
)


# =============================================================================
# REGRESSION CONSOLE SUMMARY
# =============================================================================

print()

print(
    "REGRESSION: RMSE"
)

print(
    "----------------"
)


regression_display = primary_comparison[

    [
        "scenario",
        "direct_longgcn_rmse",
        "mice3d_deep_relu_rmse",
        "conventional_rmse_gain"
    ]

].copy()


print(

    regression_display.to_string(
        index=False
    )
)


print()

print(
    "Positive conventional_rmse_gain values favor "
    "MICE3D + Deep ReLU."
)


# =============================================================================
# GROUP + MCAR THREE-STRATEGY CONSOLE SUMMARY
# =============================================================================

print()

print(
    "GROUP-SPECIFIC + MCAR: THREE-STRATEGY COMPARISON"
)

print(
    "------------------------------------------------"
)


group_display = group_mcar_plot_data[

    [
        "strategy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "rmse",
        "mae",
        "r2"
    ]

].copy()


print(

    group_display.to_string(
        index=False
    )
)


# =============================================================================
# OUTPUT LOCATIONS
# =============================================================================

print()

print(
    "Output tables saved to:"
)


print(
    STRATEGY_LEVEL_RESULTS_PATH
)


print(
    PRIMARY_COMPARISON_PATH
)


print(
    GROUP_MCAR_COMPARISON_PATH
)


print()

print(
    "Figures saved to:"
)


print(
    FIGURE_DIRECTORY
)


print()