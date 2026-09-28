"""
Run Deep ReLU Models
====================

This script trains ordinary deep ReLU prediction models on the complete or
MICE3D-completed longitudinal datasets.

The goal is to provide a non-graph neural-network comparator to LongGCN.


Conditions
----------

The Deep ReLU model is fitted to:

    complete

    mcar_to_complete

    group_specific_to_complete

    group_specific_mcar_to_complete


The structurally incomplete conditions are intentionally excluded because the
purpose of this comparator is to represent the conventional workflow:

    incomplete longitudinal data
        ↓
    MICE3D
        ↓
    complete visit vectors
        ↓
    ordinary deep ReLU model


Matched architecture
--------------------

The Deep ReLU model uses:

    5 -> 8 initial transformation

    four 8 -> 8 ordinary Linear + ReLU layers

    max pooling

    8 -> 1 output

Total trainable parameters:

    345

This exactly matches the parameter count of the one-group LongGCN model.


Training
--------

The same training utilities and hyperparameters used for LongGCN are used
here:

    Adam
    initial learning rate = 0.001
    validation-loss ReduceLROnPlateau scheduler
    scheduler factor = 0.5
    scheduler patience = 10
    minimum learning rate = 1e-6
    early-stopping patience = 100
    meaningful validation improvement = 1e-4
    best model evaluated every epoch

Classification uses:

    BCEWithLogitsLoss

Regression uses:

    MSELoss


Run from the PyTorch-Practice project root using:

    python scripts/06c_run_deep_relu_models.py
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


from src.deep_relu_models import (
    DeepReLUPredictionModel
)


from src.model_training import (
    train_model
)


from src.model_evaluation import (
    evaluate_classification_model,
    evaluate_regression_model
)


# =============================================================================
# 2. DATA DIRECTORY
# =============================================================================

LONGGCN_DATA_DIRECTORY = (

    PROJECT_ROOT

    /

    "data"

    /

    "longgcn"
)


# =============================================================================
# 3. OUTPUT DIRECTORY
# =============================================================================

RESULTS_DIRECTORY = (

    PROJECT_ROOT

    /

    "results"

    /

    "deep_relu"
)


RESULTS_DIRECTORY.mkdir(

    parents=True,

    exist_ok=True
)


COMBINED_RESULTS_PATH = (

    PROJECT_ROOT

    /

    "results"

    /

    "deep_relu_model_results.csv"
)


# =============================================================================
# 4. COMPLETED DATA CONDITIONS
# =============================================================================

# Only datasets containing complete measurement vectors at each observed
# patient time are used.

MODELING_CONDITIONS = [

    "complete",

    "mcar_to_complete",

    "group_specific_to_complete",

    "group_specific_mcar_to_complete"
]


# =============================================================================
# 5. PREDICTION TASKS
# =============================================================================

PREDICTION_TASKS = [

    "classification",

    "regression"
]


# =============================================================================
# 6. RANDOM SEED
# =============================================================================

SEED = 100


# =============================================================================
# 7. DATA LOADER SETTINGS
# =============================================================================

BATCH_SIZE = 32


NUMBER_OF_WORKERS = 0


# =============================================================================
# 8. MODEL SETTINGS
# =============================================================================

NUMBER_OF_MEASUREMENTS = 5


LATENT_DIMENSION = 8


HIDDEN_DIMENSION = 8


# The one-group LongGCN has:
#
#     2 group layers
#     +
#     2 temporal layers
#
# giving four message-passing transformations.
#
# The Deep ReLU comparator therefore uses four ordinary hidden layers.

NUMBER_OF_HIDDEN_LAYERS = 4


POOLING = "max"


# =============================================================================
# 9. TRAINING SETTINGS
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
# 10. DEVICE
# =============================================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"
)


# =============================================================================
# RESET RANDOM SEEDS
# =============================================================================

def reset_random_seeds(
    seed
):
    """
    Reset all random-number generators before one model fit.
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
    Count total and trainable parameters.
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
# LOAD PREPARED DATASET
# =============================================================================

def load_dataset(
    condition,
    task,
    split_name
):
    """
    Load one dataset prepared by script 05.

    Expected structure:

        data/
            longgcn/
                condition/
                    task/
                        train_dataset.pt
                        validation_dataset.pt
                        test_dataset.pt
    """

    dataset_path = (

        LONGGCN_DATA_DIRECTORY

        /

        condition

        /

        task

        /

        f"{split_name}_dataset.pt"
    )


    if not dataset_path.exists():

        raise FileNotFoundError(

            "Prepared LongGCN dataset was not found:\n"
            f"{dataset_path}\n\n"
            "06c reuses the completed datasets prepared by script 05."
        )


    dataset = torch.load(

        dataset_path,

        map_location="cpu",

        weights_only=False
    )


    return dataset


# =============================================================================
# CREATE DATA LOADERS
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
# GET PATIENT IDS
# =============================================================================

def get_patient_ids(
    dataset,
    number_of_patients
):
    """
    Return saved patient IDs when available.

    A sequential index is used as a fallback.
    """

    if hasattr(
        dataset,
        "patient_ids"
    ):

        patient_ids = list(
            dataset.patient_ids
        )


        if len(
            patient_ids
        ) == number_of_patients:

            return patient_ids


    return list(

        range(
            number_of_patients
        )
    )


# =============================================================================
# SAVE TRAINING HISTORY
# =============================================================================

def save_training_history(
    training_results,
    output_path
):
    """
    Save epoch-level training history.
    """

    number_of_completed_epochs = len(

        training_results[
            "training_losses"
        ]
    )


    history = pd.DataFrame({

        "epoch":
            np.arange(
                1,
                number_of_completed_epochs + 1
            ),

        "training_loss":
            training_results[
                "training_losses"
            ],

        "validation_loss":
            training_results[
                "validation_losses"
            ],

        "learning_rate":
            training_results[
                "learning_rates"
            ]
    })


    history.to_csv(

        output_path,

        index=False
    )


# =============================================================================
# CREATE MODEL
# =============================================================================

def create_model():
    """
    Create the parameter-matched Deep ReLU model.
    """

    return DeepReLUPredictionModel(

        number_of_measurements=(
            NUMBER_OF_MEASUREMENTS
        ),

        latent_dimension=(
            LATENT_DIMENSION
        ),

        hidden_dimension=(
            HIDDEN_DIMENSION
        ),

        number_of_hidden_layers=(
            NUMBER_OF_HIDDEN_LAYERS
        ),

        pooling=(
            POOLING
        )
    )


# =============================================================================
# 11. RUN EXPERIMENTS
# =============================================================================

result_rows = []


total_models = (

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
    "RUNNING DEEP RELU COMPARISON MODELS"
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
    "Completed conditions:",
    MODELING_CONDITIONS
)


print(
    "Total models:",
    total_models
)


# =============================================================================
# 12. LOOP OVER TASKS
# =============================================================================

for task in PREDICTION_TASKS:


    # =========================================================================
    # 13. LOOP OVER CONDITIONS
    # =========================================================================

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
            "DEEP RELU"
        )


        print(
            "=" * 80
        )


        # ---------------------------------------------------------------------
        # RESET RANDOM STATE
        # ---------------------------------------------------------------------

        reset_random_seeds(
            SEED
        )


        # ---------------------------------------------------------------------
        # LOAD DATA
        # ---------------------------------------------------------------------

        train_dataset = load_dataset(

            condition=condition,

            task=task,

            split_name="train"
        )


        validation_dataset = load_dataset(

            condition=condition,

            task=task,

            split_name="validation"
        )


        test_dataset = load_dataset(

            condition=condition,

            task=task,

            split_name="test"
        )


        print(
            "Training patients:",
            len(
                train_dataset
            )
        )


        print(
            "Validation patients:",
            len(
                validation_dataset
            )
        )


        print(
            "Test patients:",
            len(
                test_dataset
            )
        )


        # ---------------------------------------------------------------------
        # CREATE LOADERS
        # ---------------------------------------------------------------------

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


        # ---------------------------------------------------------------------
        # CREATE MODEL
        # ---------------------------------------------------------------------

        model = create_model()


        parameter_counts = (
            count_model_parameters(
                model
            )
        )


        print(
            "Total parameters:",
            parameter_counts[
                "total_parameters"
            ]
        )


        print(
            "Trainable parameters:",
            parameter_counts[
                "trainable_parameters"
            ]
        )


        # Exact parameter matching is part of the experimental design.

        if (
            parameter_counts[
                "trainable_parameters"
            ]
            !=
            345
        ):

            raise RuntimeError(

                "Deep ReLU parameter count is not 345. "
                "The comparator architecture is no longer matched to "
                "the one-group LongGCN model."
            )


        # ---------------------------------------------------------------------
        # TRAIN MODEL
        # ---------------------------------------------------------------------

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


        trained_model = training_results[
            "model"
        ]


        # ---------------------------------------------------------------------
        # TEST EVALUATION
        # ---------------------------------------------------------------------

        if task == "classification":

            evaluation_results = (
                evaluate_classification_model(

                    model=trained_model,

                    data_loader=test_loader,

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

                    data_loader=test_loader,

                    device=DEVICE
                )
            )


        # ---------------------------------------------------------------------
        # CREATE OUTPUT DIRECTORY
        # ---------------------------------------------------------------------

        experiment_directory = (

            RESULTS_DIRECTORY

            /

            task

            /

            condition
        )


        experiment_directory.mkdir(

            parents=True,

            exist_ok=True
        )


        # ---------------------------------------------------------------------
        # SAVE BEST MODEL
        # ---------------------------------------------------------------------

        torch.save(

            trained_model.state_dict(),

            experiment_directory

            /

            "best_model_state.pt"
        )


        # ---------------------------------------------------------------------
        # SAVE TRAINING HISTORY
        # ---------------------------------------------------------------------

        save_training_history(

            training_results=(
                training_results
            ),

            output_path=(

                experiment_directory

                /

                "training_history.csv"
            )
        )


        # ---------------------------------------------------------------------
        # SAVE CHECKPOINT HISTORY
        # ---------------------------------------------------------------------

        pd.DataFrame(

            training_results[
                "checkpoint_history"
            ]

        ).to_csv(

            experiment_directory

            /

            "checkpoint_history.csv",

            index=False
        )


        # ---------------------------------------------------------------------
        # SAVE TEST PREDICTIONS
        # ---------------------------------------------------------------------

        number_of_test_patients = len(

            evaluation_results[
                "true_outcomes"
            ]
        )


        patient_ids = get_patient_ids(

            dataset=test_dataset,

            number_of_patients=(
                number_of_test_patients
            )
        )


        if task == "classification":

            prediction_data = pd.DataFrame({

                "patient_id":
                    patient_ids,

                "true_outcome":
                    evaluation_results[
                        "true_outcomes"
                    ],

                "predicted_logit":
                    evaluation_results[
                        "predicted_logits"
                    ],

                "predicted_probability":
                    evaluation_results[
                        "predicted_probabilities"
                    ],

                "predicted_class":
                    evaluation_results[
                        "predicted_classes"
                    ]
            })


        else:

            prediction_data = pd.DataFrame({

                "patient_id":
                    patient_ids,

                "true_outcome":
                    evaluation_results[
                        "true_outcomes"
                    ],

                "predicted_outcome":
                    evaluation_results[
                        "predicted_outcomes"
                    ]
            })


        prediction_data.to_csv(

            experiment_directory

            /

            "test_predictions.csv",

            index=False
        )


        # ---------------------------------------------------------------------
        # SAVE EXPERIMENT SETTINGS
        # ---------------------------------------------------------------------

        experiment_settings = {

            "architecture":
                "deep_relu",

            "condition":
                condition,

            "task":
                task,

            "seed":
                SEED,

            "number_of_measurements":
                NUMBER_OF_MEASUREMENTS,

            "latent_dimension":
                LATENT_DIMENSION,

            "hidden_dimension":
                HIDDEN_DIMENSION,

            "number_of_hidden_layers":
                NUMBER_OF_HIDDEN_LAYERS,

            "pooling":
                POOLING,

            "total_parameters":
                parameter_counts[
                    "total_parameters"
                ],

            "trainable_parameters":
                parameter_counts[
                    "trainable_parameters"
                ],

            "number_of_epochs_maximum":
                NUMBER_OF_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "scheduler_factor":
                SCHEDULER_FACTOR,

            "scheduler_patience":
                SCHEDULER_PATIENCE,

            "minimum_learning_rate":
                MINIMUM_LEARNING_RATE,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "minimum_validation_improvement":
                MINIMUM_VALIDATION_IMPROVEMENT,

            "checkpoint_interval":
                CHECKPOINT_INTERVAL,

            "classification_threshold":
                CLASSIFICATION_THRESHOLD
        }


        with open(

            experiment_directory

            /

            "experiment_settings.json",

            "w",

            encoding="utf-8"

        ) as settings_file:

            json.dump(

                experiment_settings,

                settings_file,

                indent=4
            )


        # ---------------------------------------------------------------------
        # CREATE RESULT SUMMARY ROW
        # ---------------------------------------------------------------------

        result_row = {

            "architecture":
                "deep_relu",

            "condition":
                condition,

            "task":
                task,

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
                training_results.get(

                    "epochs_completed",

                    len(
                        training_results[
                            "training_losses"
                        ]
                    )
                ),

            "stopped_early":
                training_results.get(

                    "stopped_early",

                    False
                ),

            "test_loss":
                evaluation_results[
                    "loss"
                ]
        }


        # ---------------------------------------------------------------------
        # ADD TASK-SPECIFIC METRICS
        # ---------------------------------------------------------------------

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
                    ],

                "true_positives":
                    evaluation_results[
                        "true_positives"
                    ],

                "true_negatives":
                    evaluation_results[
                        "true_negatives"
                    ],

                "false_positives":
                    evaluation_results[
                        "false_positives"
                    ],

                "false_negatives":
                    evaluation_results[
                        "false_negatives"
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


        # ---------------------------------------------------------------------
        # SAVE COMBINED RESULTS AFTER EVERY MODEL
        # ---------------------------------------------------------------------

        pd.DataFrame(

            result_rows

        ).to_csv(

            COMBINED_RESULTS_PATH,

            index=False
        )


        # ---------------------------------------------------------------------
        # PRINT RESULT
        # ---------------------------------------------------------------------

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


        # ---------------------------------------------------------------------
        # CLEAN UP
        # ---------------------------------------------------------------------

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
# 14. FINAL SUMMARY
# =============================================================================

results = pd.DataFrame(
    result_rows
)


results.to_csv(

    COMBINED_RESULTS_PATH,

    index=False
)


print()

print(
    "=" * 80
)

print(
    "DEEP RELU EXPERIMENT COMPLETE"
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
    COMBINED_RESULTS_PATH
)


print()