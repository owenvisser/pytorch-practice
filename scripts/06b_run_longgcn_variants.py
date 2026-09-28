"""
Run LongGCN Temporal-Decay Variants
===================================

This script trains LongGCN prediction models across the temporal-decay
variants prepared by:

    scripts/05b_prepare_longgcn_variants.py

The current experiment varies:

    temporal weighting:
        exponential

    decay parameter d:
        0.25
        0.50
        0.75
        1.00
        2.00
        5.00

while keeping all other model and training settings constant.

For every value of d, all eight missing-data / imputation conditions are
evaluated for both:

    classification
    regression

The purpose is to determine whether conclusions about the missing-data
strategy are robust to the assumed temporal-decay scale.


Run from the project root using:

    python scripts/06b_run_longgcn_variants.py
"""


from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader


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


from longgcn.data import (
    collate_longgcn
)


from src.longgcn_models import (
    LongGCNPredictionModel
)


from src.model_training import (
    train_model
)


from src.model_evaluation import (
    evaluate_classification_model,
    evaluate_regression_model
)


# =============================================================================
# 2. INPUT / OUTPUT DIRECTORIES
# =============================================================================

VARIANT_DATA_DIRECTORY = (

    PROJECT_ROOT
    /
    "data"
    /
    "longgcn_variants"
)


RESULTS_DIRECTORY = (

    PROJECT_ROOT
    /
    "results"
    /
    "temporal_variants"
)


RESULTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 3. TEMPORAL EXPERIMENT
# =============================================================================

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
# 4. CONDITIONS
# =============================================================================

MODELING_CONDITIONS = [

    "complete",

    "mcar",

    "mcar_to_complete",

    "group_specific",

    "group_specific_to_complete",

    "group_specific_mcar",

    "group_specific_mcar_to_group_specific",

    "group_specific_mcar_to_complete"
]


PREDICTION_TASKS = [

    "classification",

    "regression"
]


# =============================================================================
# 5. RANDOM SEED
# =============================================================================

SEED = 100


# =============================================================================
# 6. DATA LOADER SETTINGS
# =============================================================================

BATCH_SIZE = 32


NUMBER_OF_WORKERS = 0


# =============================================================================
# 7. MODEL SETTINGS
# =============================================================================

NUMBER_OF_MEASUREMENTS = 5


LATENT_DIMENSION = 8


HIDDEN_DIMENSION = 8


NUMBER_OF_GROUP_LAYERS = 2


NUMBER_OF_TEMPORAL_LAYERS = 2


GROUP_AGGREGATION = "sum"


TEMPORAL_AGGREGATION = "sum"


PARAMETER_SHARING = "group"


POOLING = "max"


# =============================================================================
# 8. TRAINING SETTINGS
# =============================================================================

NUMBER_OF_EPOCHS = 500


LEARNING_RATE = 0.001


WEIGHT_DECAY = 0.0


SCHEDULER_FACTOR = 0.5


SCHEDULER_PATIENCE = 10


MINIMUM_LEARNING_RATE = 1e-6


EARLY_STOPPING_PATIENCE = 100


MINIMUM_VALIDATION_IMPROVEMENT = 1e-4


CHECKPOINT_INTERVAL = 1


PRINT_INTERVAL = 20


CLASSIFICATION_THRESHOLD = 0.5


# =============================================================================
# 9. DEVICE
# =============================================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"
)


# =============================================================================
# FORMAT d
# =============================================================================

def format_decay_parameter(
    decay_parameter
):
    """
    Convert d to the directory label used by script 05b.
    """

    text = f"{decay_parameter:g}"

    text = text.replace(
        ".",
        "p"
    )


    return f"d_{text}"


# =============================================================================
# RESET RANDOM STATE
# =============================================================================

def reset_random_seeds(
    seed
):
    """
    Reset random-number generators before every model fit.
    """

    random.seed(
        seed
    )


    np.random.seed(
        seed
    )


    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =============================================================================
# COUNT MODEL PARAMETERS
# =============================================================================

def count_model_parameters(
    model
):
    """
    Count total and trainable model parameters.
    """

    total_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()
    )


    trainable_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad
    )


    return {

        "total_parameters":
            total_parameters,

        "trainable_parameters":
            trainable_parameters
    }


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_variant_dataset(
    decay_parameter,
    condition,
    task,
    split_name
):
    """
    Load one temporal-variant LongGCNTorchDataset.
    """

    decay_label = (
        format_decay_parameter(
            decay_parameter
        )
    )


    dataset_path = (

        VARIANT_DATA_DIRECTORY

        /

        TEMPORAL_WEIGHTING

        /

        decay_label

        /

        condition

        /

        task

        /

        f"{split_name}_dataset.pt"
    )


    if not dataset_path.exists():

        raise FileNotFoundError(

            "Variant dataset was not found:\n"
            f"{dataset_path}"
        )


    return torch.load(

        dataset_path,

        map_location="cpu",

        weights_only=False
    )


# =============================================================================
# CREATE LOADERS
# =============================================================================

def create_data_loaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    seed
):
    """
    Create train, validation, and test DataLoaders.
    """

    generator = torch.Generator()


    generator.manual_seed(
        seed
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUMBER_OF_WORKERS,

        collate_fn=collate_longgcn,

        generator=generator,

        drop_last=False
    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUMBER_OF_WORKERS,

        collate_fn=collate_longgcn,

        drop_last=False
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUMBER_OF_WORKERS,

        collate_fn=collate_longgcn,

        drop_last=False
    )


    return (
        train_loader,
        validation_loader,
        test_loader
    )


# =============================================================================
# CREATE MODEL
# =============================================================================

def create_model(
    group_names
):
    """
    Construct the common LongGCN prediction architecture.
    """

    return LongGCNPredictionModel(

        number_of_measurements=(
            NUMBER_OF_MEASUREMENTS
        ),

        group_names=(
            group_names
        ),

        latent_dimension=(
            LATENT_DIMENSION
        ),

        hidden_dimension=(
            HIDDEN_DIMENSION
        ),

        number_of_group_layers=(
            NUMBER_OF_GROUP_LAYERS
        ),

        number_of_temporal_layers=(
            NUMBER_OF_TEMPORAL_LAYERS
        ),

        group_aggregation=(
            GROUP_AGGREGATION
        ),

        temporal_aggregation=(
            TEMPORAL_AGGREGATION
        ),

        parameter_sharing=(
            PARAMETER_SHARING
        ),

        pooling=(
            POOLING
        )
    )


# =============================================================================
# 10. RUN EXPERIMENT GRID
# =============================================================================

result_rows = []


total_models = (

    len(
        DECAY_PARAMETERS
    )

    *

    len(
        MODELING_CONDITIONS
    )

    *

    len(
        PREDICTION_TASKS
    )
)


model_number = 0


print()

print(
    "=" * 80
)

print(
    "RUNNING LONGGCN TEMPORAL-DECAY VARIANTS"
)

print(
    "=" * 80
)


print()

print(
    "Device:",
    DEVICE
)


print(
    "Temporal weighting:",
    TEMPORAL_WEIGHTING
)


print(
    "d values:",
    DECAY_PARAMETERS
)


print(
    "Total model fits:",
    total_models
)


# =============================================================================
# 11. LOOP OVER d
# =============================================================================

for decay_parameter in DECAY_PARAMETERS:

    decay_label = (
        format_decay_parameter(
            decay_parameter
        )
    )


    # =========================================================================
    # 12. LOOP OVER TASK
    # =========================================================================

    for task in PREDICTION_TASKS:

        # =====================================================================
        # 13. LOOP OVER CONDITION
        # =====================================================================

        for condition in MODELING_CONDITIONS:

            model_number += 1


            print()

            print(
                "=" * 80
            )


            print(

                f"MODEL {model_number} OF {total_models}"
            )


            print(

                f"{task.upper()} | "
                f"{condition} | "
                f"d={decay_parameter:g}"
            )


            print(
                "=" * 80
            )


            # -----------------------------------------------------------------
            # RESET RANDOM NUMBERS
            # -----------------------------------------------------------------

            reset_random_seeds(
                SEED
            )


            # -----------------------------------------------------------------
            # LOAD DATA
            # -----------------------------------------------------------------

            train_dataset = (
                load_variant_dataset(

                    decay_parameter=(
                        decay_parameter
                    ),

                    condition=condition,

                    task=task,

                    split_name="train"
                )
            )


            validation_dataset = (
                load_variant_dataset(

                    decay_parameter=(
                        decay_parameter
                    ),

                    condition=condition,

                    task=task,

                    split_name="validation"
                )
            )


            test_dataset = (
                load_variant_dataset(

                    decay_parameter=(
                        decay_parameter
                    ),

                    condition=condition,

                    task=task,

                    split_name="test"
                )
            )


            group_names = list(
                train_dataset.group_names
            )


            # -----------------------------------------------------------------
            # LOADERS
            # -----------------------------------------------------------------

            (
                train_loader,
                validation_loader,
                test_loader

            ) = create_data_loaders(

                train_dataset=(
                    train_dataset
                ),

                validation_dataset=(
                    validation_dataset
                ),

                test_dataset=(
                    test_dataset
                ),

                seed=SEED
            )


            # -----------------------------------------------------------------
            # MODEL
            # -----------------------------------------------------------------

            model = create_model(

                group_names=(
                    group_names
                )
            )


            parameter_counts = (
                count_model_parameters(
                    model
                )
            )


            print(
                "Groups:",
                group_names
            )


            print(
                "Parameters:",
                parameter_counts[
                    "trainable_parameters"
                ]
            )


            # -----------------------------------------------------------------
            # TRAIN
            # -----------------------------------------------------------------

            training_results = train_model(

                model=model,

                train_loader=train_loader,

                validation_loader=(
                    validation_loader
                ),

                task=task,

                device=DEVICE,

                number_of_epochs=(
                    NUMBER_OF_EPOCHS
                ),

                learning_rate=(
                    LEARNING_RATE
                ),

                weight_decay=(
                    WEIGHT_DECAY
                ),

                scheduler_factor=(
                    SCHEDULER_FACTOR
                ),

                scheduler_patience=(
                    SCHEDULER_PATIENCE
                ),

                minimum_learning_rate=(
                    MINIMUM_LEARNING_RATE
                ),

                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                ),

                minimum_validation_improvement=(
                    MINIMUM_VALIDATION_IMPROVEMENT
                ),

                checkpoint_interval=(
                    CHECKPOINT_INTERVAL
                ),

                print_interval=(
                    PRINT_INTERVAL
                )
            )


            trained_model = (

                training_results[
                    "model"
                ]
            )


            # -----------------------------------------------------------------
            # EVALUATE
            # -----------------------------------------------------------------

            if task == "classification":

                evaluation_results = (
                    evaluate_classification_model(

                        model=trained_model,

                        data_loader=(
                            test_loader
                        ),

                        device=DEVICE,

                        probability_threshold=(
                            CLASSIFICATION_THRESHOLD
                        )
                    )
                )


            else:

                evaluation_results = (
                    evaluate_regression_model(

                        model=trained_model,

                        data_loader=(
                            test_loader
                        ),

                        device=DEVICE
                    )
                )


            # -----------------------------------------------------------------
            # RESULT ROW
            # -----------------------------------------------------------------

            result_row = {

                "temporal_weighting":
                    TEMPORAL_WEIGHTING,

                "decay_parameter":
                    decay_parameter,

                "decay_label":
                    decay_label,

                "condition":
                    condition,

                "task":
                    task,

                "number_of_groups":
                    len(
                        group_names
                    ),

                "total_parameters":
                    parameter_counts[
                        "total_parameters"
                    ],

                "trainable_parameters":
                    parameter_counts[
                        "trainable_parameters"
                    ],

                "best_epoch":
                    training_results[
                        "best_epoch"
                    ],

                "best_validation_loss":
                    training_results[
                        "best_validation_loss"
                    ],

                "epochs_completed":
                    training_results[
                        "epochs_completed"
                    ],

                "stopped_early":
                    training_results[
                        "stopped_early"
                    ],

                "test_loss":
                    evaluation_results[
                        "loss"
                    ]
            }


            if task == "classification":

                result_row.update({

                    "accuracy":
                        evaluation_results[
                            "accuracy"
                        ],

                    "balanced_accuracy":
                        evaluation_results[
                            "balanced_accuracy"
                        ],

                    "sensitivity":
                        evaluation_results[
                            "sensitivity"
                        ],

                    "specificity":
                        evaluation_results[
                            "specificity"
                        ]
                })


            else:

                result_row.update({

                    "mse":
                        evaluation_results[
                            "mse"
                        ],

                    "rmse":
                        evaluation_results[
                            "rmse"
                        ],

                    "mae":
                        evaluation_results[
                            "mae"
                        ],

                    "r2":
                        evaluation_results[
                            "r2"
                        ]
                })


            result_rows.append(
                result_row
            )


            # -----------------------------------------------------------------
            # SAVE RESULTS AS WE GO
            # -----------------------------------------------------------------

            # Saving after every fit protects the sensitivity study if execution
            # is interrupted before all models are complete.

            current_results = pd.DataFrame(
                result_rows
            )


            current_results.to_csv(

                RESULTS_DIRECTORY

                /

                "temporal_variant_results.csv",

                index=False
            )


            print()

            print(
                "Best epoch:",
                training_results[
                    "best_epoch"
                ]
            )


            if task == "classification":

                print(
                    "Balanced accuracy:",
                    round(
                        evaluation_results[
                            "balanced_accuracy"
                        ],
                        6
                    )
                )


            else:

                print(
                    "RMSE:",
                    round(
                        evaluation_results[
                            "rmse"
                        ],
                        6
                    )
                )


            # -----------------------------------------------------------------
            # CLEAN UP
            # -----------------------------------------------------------------

            del model
            del trained_model
            del training_results
            del evaluation_results

            del train_loader
            del validation_loader
            del test_loader

            del train_dataset
            del validation_dataset
            del test_dataset


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


# =============================================================================
# 14. FINAL RESULTS
# =============================================================================

results = pd.DataFrame(
    result_rows
)


results_path = (

    RESULTS_DIRECTORY

    /

    "temporal_variant_results.csv"
)


results.to_csv(

    results_path,

    index=False
)


print()

print(
    "=" * 80
)

print(
    "TEMPORAL VARIANT EXPERIMENT COMPLETE"
)

print(
    "=" * 80
)


print()

print(
    "Models fit:",
    len(
        results
    )
)


print(
    "Results saved to:"
)

print(
    results_path
)


print()