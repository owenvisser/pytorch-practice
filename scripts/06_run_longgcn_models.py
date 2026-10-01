"""
Run LongGCN Simulation Models
=============================

This script trains and evaluates the LongGCN prediction models for all
simulation-study data conditions prepared by:

    scripts/05_prepare_longgcn_datasets.py

Eight data conditions are evaluated:

    1. complete

    2. mcar

    3. mcar_to_complete

    4. group_specific

    5. group_specific_to_complete

    6. group_specific_mcar

    7. group_specific_mcar_to_complete

    8. group_specific_mcar_to_group_specific


Each condition is evaluated for two prediction tasks:

    classification
    regression


Each experiment uses:

    training dataset
        for model optimization

    validation dataset
        for learning-rate scheduling and best-model selection

    test dataset
        only for final evaluation after training is complete


The model architecture is held constant wherever possible.

Complete-like conditions contain one measurement group:

    all_measurements

Group-structured conditions contain two measurement groups:

    group_1
    group_2

Consequently, models using group-specific parameters may have different
numbers of learnable parameters. Parameter counts are therefore recorded
with the model results.


For every condition/task combination this script saves:

    best_model_state.pt
        State dictionary for the retained best-validation model.

    training_history.csv
        Training loss, validation loss, and learning rate for every epoch.

    checkpoint_history.csv
        Validation information used during model selection.

    test_predictions.csv
        Patient-level test outcomes and model predictions.

    experiment_settings.json
        Model, training, and computational settings.

Task-level summary files are also created:

    results/classification/model_results.csv
    results/regression/model_results.csv

and a combined summary:

    results/longgcn_model_results.csv


Run from the PyTorch-Practice project root using:

    python scripts/06_run_longgcn_models.py
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
# 2. INPUT DIRECTORY
# =============================================================================

LONGGCN_DATA_DIRECTORY = (

    PROJECT_ROOT

    /

    "data"

    /

    "longgcn"
)


# =============================================================================
# 3. RESULTS DIRECTORY
# =============================================================================

RESULTS_DIRECTORY = (

    PROJECT_ROOT

    /

    "results"
)


RESULTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

# =============================================================================
# 4. MODELING CONDITIONS
# =============================================================================

MODELING_CONDITIONS = [

    "complete",

    "mcar",

    "mcar_to_complete",

    "group_specific",

    "group_specific_to_complete",

    "group_specific_mcar",

    "group_specific_mcar_to_complete",

    "group_specific_mcar_to_group_specific"
]

# =============================================================================
# 5. PREDICTION TASKS
# =============================================================================

PREDICTION_TASKS = [

    "classification",

    "regression"
]

# =============================================================================
# 6. REPRODUCIBILITY SETTINGS
# =============================================================================

SEED = 100

# =============================================================================
# 7. DATA LOADER SETTINGS
# =============================================================================

BATCH_SIZE = 50

# Windows multiprocessing can introduce unnecessary complications when the
# datasets are already precomputed in memory.

NUMBER_OF_WORKERS = 0

# =============================================================================
# 8. MODEL ARCHITECTURE SETTINGS
# =============================================================================

NUMBER_OF_MEASUREMENTS = 5

LATENT_DIMENSION = 8

HIDDEN_DIMENSION = 8

NUMBER_OF_GROUP_LAYERS = 2

NUMBER_OF_TEMPORAL_LAYERS = 2

GROUP_AGGREGATION = "sum"

TEMPORAL_AGGREGATION = "sum"

GROUP_ACTIVATIONS = ["relu"] * NUMBER_OF_GROUP_LAYERS

TEMPORAL_ACTIVATIONS = ["relu"] * NUMBER_OF_TEMPORAL_LAYERS

# Each designed group receives its own transformation parameters.
#
# Therefore two-group LongGCN models will contain more learnable parameters
# than one-group models. The exact counts are saved with the results.

PARAMETER_SHARING = "group"


POOLING = "max"


# =============================================================================
# 9. TRAINING SETTINGS
# =============================================================================

NUMBER_OF_EPOCHS = 1000

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 10

MINIMUM_LEARNING_RATE = 1e-6

EARLY_STOPPING_PATIENCE = 100

MINIMUM_VALIDATION_IMPROVEMENT = 1e-4

# Validation loss is already calculated every epoch by train_model().
#
# Using a checkpoint interval of 1 means that the retained model is truly the
# epoch with the lowest validation loss, rather than only the best epoch among
# 20, 40, 60, ...

CHECKPOINT_INTERVAL = 1

# Progress is still printed only every 20 epochs.

PRINT_INTERVAL = 20

# =============================================================================
# 10. CLASSIFICATION SETTINGS
# =============================================================================

CLASSIFICATION_THRESHOLD = 0.5

# =============================================================================
# 11. SELECT COMPUTING DEVICE
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
    Reset random-number generators before each model experiment.

    This helps ensure that comparable model architectures begin from the same
    random state rather than inheriting a different state simply because one
    experiment happened to run earlier in the script.
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
    Count total and trainable parameters in a PyTorch model.
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
# LOAD LONGGCN DATASET
# =============================================================================

def load_longgcn_dataset(
    condition,
    task,
    split_name
):
    """
    Load one saved LongGCNTorchDataset.
    """

    dataset_path = (LONGGCN_DATA_DIRECTORY/condition/task/f"{split_name}_dataset.pt")

    if not dataset_path.exists():
        raise FileNotFoundError(
            "LongGCN dataset was not found:\n"
            f"{dataset_path}"
        )

    # LongGCNTorchDataset is a custom Python object, so weights_only=False
    # must be used when loading it.

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
    Create reproducible LongGCN DataLoaders.
    """

    # A dedicated generator controls the training-data shuffle.

    training_generator = torch.Generator()


    training_generator.manual_seed(
        seed
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_WORKERS,
        collate_fn=collate_longgcn,
        generator=training_generator,
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
# VALIDATE DATASET COMPATIBILITY
# =============================================================================

def validate_dataset_compatibility(
    train_dataset,
    validation_dataset,
    test_dataset,
    condition,
    task
):
    """
    Verify that train, validation, and test datasets use identical group
    definitions.
    """

    train_groups = list(
        train_dataset.group_names
    )


    validation_groups = list(
        validation_dataset.group_names
    )


    test_groups = list(
        test_dataset.group_names
    )


    if train_groups != validation_groups:

        raise ValueError(

            f"{condition}, {task}: "
            "training and validation group definitions do not match."
        )


    if train_groups != test_groups:

        raise ValueError(

            f"{condition}, {task}: "
            "training and test group definitions do not match."
        )


    return train_groups


# =============================================================================
# CREATE MODEL
# =============================================================================

def create_model(
    group_names
):
    """
    Create one LongGCN prediction model using the common simulation settings.
    """

    model = LongGCNPredictionModel(

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

        group_activations=(
            GROUP_ACTIVATIONS
        ),

        temporal_activations=(
            TEMPORAL_ACTIVATIONS
        ),

        parameter_sharing=(
            PARAMETER_SHARING
        ),

        pooling=(
            POOLING
        )
    )


    return model


# =============================================================================
# SAVE TRAINING HISTORY
# =============================================================================

def save_training_history(
    training_results,
    output_path
):
    """
    Save one row per training epoch.
    """

    number_of_epochs_completed = len(

        training_results[
            "training_losses"
        ]
    )


    training_history = pd.DataFrame({

        "epoch":
            np.arange(
                1,
                number_of_epochs_completed + 1
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


    training_history.to_csv(

        output_path,

        index=False
    )


# =============================================================================
# SAVE CHECKPOINT HISTORY
# =============================================================================

def save_checkpoint_history(
    training_results,
    output_path
):
    """
    Save validation-loss checkpoint information.
    """

    checkpoint_history = pd.DataFrame(

        training_results[
            "checkpoint_history"
        ]
    )


    checkpoint_history.to_csv(

        output_path,

        index=False
    )


# =============================================================================
# SAVE TEST PREDICTIONS
# =============================================================================

def save_test_predictions(
    task,
    test_dataset,
    evaluation_results,
    output_path
):
    """
    Save patient-level test outcomes and predictions.
    """

    patient_ids = list(
        test_dataset.patient_ids
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


    elif task == "regression":

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


    else:

        raise ValueError(
            "task must be either "
            "'classification' or 'regression'."
        )


    if len(
        prediction_data
    ) != len(
        test_dataset
    ):

        raise RuntimeError(
            "The number of saved predictions does not match "
            "the number of patients in the test dataset."
        )


    prediction_data.to_csv(

        output_path,

        index=False
    )


# =============================================================================
# SAVE BEST MODEL STATE
# =============================================================================

def save_model_state(
    model,
    output_path
):
    """
    Save the trained model state dictionary on CPU.
    """

    cpu_state_dict = {

        parameter_name:
            parameter_value
            .detach()
            .cpu()

        for (
            parameter_name,
            parameter_value
        ) in model.state_dict().items()
    }


    torch.save(

        cpu_state_dict,

        output_path
    )


# =============================================================================
# INITIALIZE RESULT STORAGE
# =============================================================================

all_results = []


task_results = {

    "classification":
        [],

    "regression":
        []
}


# =============================================================================
# 12. DISPLAY STUDY SETTINGS
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "RUNNING LONGGCN SIMULATION MODELS"
)

print(
    "=" * 80
)


print()

print(
    "Device:",
    DEVICE
)


if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(
            DEVICE
        )
    )


print()

print(
    "Conditions:",
    len(
        MODELING_CONDITIONS
    )
)


print(
    "Prediction tasks:",
    len(
        PREDICTION_TASKS
    )
)


print(
    "Total model fits:",
    (
        len(
            MODELING_CONDITIONS
        )
        *
        len(
            PREDICTION_TASKS
        )
    )
)


print()

print(
    "Epochs:",
    NUMBER_OF_EPOCHS
)


print(
    "Batch size:",
    BATCH_SIZE
)


print(
    "Initial learning rate:",
    LEARNING_RATE
)


print(
    "Pooling:",
    POOLING
)


print(
    "Group aggregation:",
    GROUP_AGGREGATION
)


print(
    "Temporal aggregation:",
    TEMPORAL_AGGREGATION
)


print(
    "Parameter sharing:",
    PARAMETER_SHARING
)


# =============================================================================
# 13. RUN EACH PREDICTION TASK
# =============================================================================

for task in PREDICTION_TASKS:

    task_result_directory = (

        RESULTS_DIRECTORY

        /

        task
    )


    task_result_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    print()

    print(
        "#" * 80
    )

    print(
        f"TASK: {task.upper()}"
    )

    print(
        "#" * 80
    )


    # =========================================================================
    # 14. RUN EACH DATA CONDITION
    # =========================================================================

    for condition in MODELING_CONDITIONS:

        print()

        print(
            "=" * 80
        )

        print(
            f"{task.upper()} | {condition}"
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
        # CREATE EXPERIMENT RESULT DIRECTORY
        # ---------------------------------------------------------------------

        experiment_directory = (

            task_result_directory

            /

            condition
        )


        experiment_directory.mkdir(
            parents=True,
            exist_ok=True
        )


        # ---------------------------------------------------------------------
        # LOAD TRAIN / VALIDATION / TEST DATASETS
        # ---------------------------------------------------------------------

        train_dataset = (
            load_longgcn_dataset(

                condition=condition,

                task=task,

                split_name="train"
            )
        )


        validation_dataset = (
            load_longgcn_dataset(

                condition=condition,

                task=task,

                split_name="validation"
            )
        )


        test_dataset = (
            load_longgcn_dataset(

                condition=condition,

                task=task,

                split_name="test"
            )
        )


        # ---------------------------------------------------------------------
        # VALIDATE GROUP DEFINITIONS
        # ---------------------------------------------------------------------

        group_names = (
            validate_dataset_compatibility(

                train_dataset=(
                    train_dataset
                ),

                validation_dataset=(
                    validation_dataset
                ),

                test_dataset=(
                    test_dataset
                ),

                condition=condition,

                task=task
            )
        )


        number_of_groups = len(
            group_names
        )


        print(
            "Measurement groups:",
            group_names
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
        # CREATE DATA LOADERS
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

        model = create_model(
            group_names=group_names
        )


        # ---------------------------------------------------------------------
        # COUNT MODEL PARAMETERS
        # ---------------------------------------------------------------------

        parameter_counts = (
            count_model_parameters(
                model
            )
        )


        total_parameters = (
            parameter_counts[
                "total_parameters"
            ]
        )


        trainable_parameters = (
            parameter_counts[
                "trainable_parameters"
            ]
        )


        print(
            "Total model parameters:",
            total_parameters
        )


        print(
            "Trainable parameters:",
            trainable_parameters
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

            checkpoint_interval=(
                CHECKPOINT_INTERVAL
            ),

            print_interval=(
                PRINT_INTERVAL
            ),

            early_stopping_patience=(
                EARLY_STOPPING_PATIENCE
            ),

            minimum_validation_improvement=(
                MINIMUM_VALIDATION_IMPROVEMENT
            )
        )


        trained_model = (
            training_results[
                "model"
            ]
        )


        best_epoch = (
            training_results[
                "best_epoch"
            ]
        )


        best_validation_loss = (
            training_results[
                "best_validation_loss"
            ]
        )


        epochs_completed = (
            training_results[
                "epochs_completed"
            ]
        )

        stopped_early = (
            training_results[
                "stopped_early"
            ]
        )

        print()

        print(
            "Best epoch:",
            best_epoch
        )


        print(
            "Best validation loss:",
            round(
                best_validation_loss,
                6
            )
        )


        # ---------------------------------------------------------------------
        # EVALUATE TEST DATA
        # ---------------------------------------------------------------------

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


        elif task == "regression":

            evaluation_results = (
                evaluate_regression_model(

                    model=trained_model,

                    data_loader=(
                        test_loader
                    ),

                    device=DEVICE
                )
            )


        else:

            raise ValueError(
                "Unknown prediction task."
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

        save_checkpoint_history(

            training_results=(
                training_results
            ),

            output_path=(

                experiment_directory

                /

                "checkpoint_history.csv"
            )
        )


        # ---------------------------------------------------------------------
        # SAVE PATIENT-LEVEL TEST PREDICTIONS
        # ---------------------------------------------------------------------

        save_test_predictions(

            task=task,

            test_dataset=(
                test_dataset
            ),

            evaluation_results=(
                evaluation_results
            ),

            output_path=(

                experiment_directory

                /

                "test_predictions.csv"
            )
        )


        # ---------------------------------------------------------------------
        # SAVE BEST MODEL STATE
        # ---------------------------------------------------------------------

        save_model_state(

            model=trained_model,

            output_path=(

                experiment_directory

                /

                "best_model_state.pt"
            )
        )


        # ---------------------------------------------------------------------
        # SAVE EXPERIMENT SETTINGS
        # ---------------------------------------------------------------------

        experiment_settings = {

            "condition":
                condition,

            "task":
                task,

            "seed":
                SEED,

            "device":
                str(
                    DEVICE
                ),

            "batch_size":
                BATCH_SIZE,

            "number_of_workers":
                NUMBER_OF_WORKERS,

            "number_of_measurements":
                NUMBER_OF_MEASUREMENTS,

            "group_names":
                group_names,

            "number_of_groups":
                number_of_groups,

            "latent_dimension":
                LATENT_DIMENSION,

            "hidden_dimension":
                HIDDEN_DIMENSION,

            "number_of_group_layers":
                NUMBER_OF_GROUP_LAYERS,

            "number_of_temporal_layers":
                NUMBER_OF_TEMPORAL_LAYERS,

            "group_aggregation":
                GROUP_AGGREGATION,

            "temporal_aggregation":
                TEMPORAL_AGGREGATION,

            "parameter_sharing":
                PARAMETER_SHARING,

            "pooling":
                POOLING,

            "total_parameters":
                total_parameters,

            "trainable_parameters":
                trainable_parameters,

            "number_of_epochs":
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

            "checkpoint_interval":
                CHECKPOINT_INTERVAL,

            "print_interval":
                PRINT_INTERVAL,

            "classification_threshold":
                (
                    CLASSIFICATION_THRESHOLD

                    if task == "classification"

                    else None
                ),

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "minimum_validation_improvement":
                MINIMUM_VALIDATION_IMPROVEMENT,

            "epochs_completed":
                epochs_completed,

            "stopped_early":
                stopped_early,

            "best_epoch":
                best_epoch,

            "best_validation_loss":
                best_validation_loss
        }


        settings_path = (

            experiment_directory

            /

            "experiment_settings.json"
        )


        with open(
            settings_path,
            "w",
            encoding="utf-8"
        ) as settings_file:

            json.dump(

                experiment_settings,

                settings_file,

                indent=4
            )


        # ---------------------------------------------------------------------
        # CREATE COMMON RESULT SUMMARY
        # ---------------------------------------------------------------------

        result_summary = {

            "condition":
                condition,

            "task":
                task,

            "number_of_groups":
                number_of_groups,

            "group_names":
                ", ".join(
                    group_names
                ),

            "total_parameters":
                total_parameters,

            "trainable_parameters":
                trainable_parameters,

            "epochs_completed":
                epochs_completed,

            "stopped_early":
                stopped_early,

            "best_epoch":
                best_epoch,

            "best_validation_loss":
                best_validation_loss,

            "test_loss":
                evaluation_results[
                    "loss"
                ]
        }


        # ---------------------------------------------------------------------
        # ADD CLASSIFICATION METRICS
        # ---------------------------------------------------------------------

        if task == "classification":

            result_summary.update({

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


        # ---------------------------------------------------------------------
        # ADD REGRESSION METRICS
        # ---------------------------------------------------------------------

        if task == "regression":

            result_summary.update({

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


        all_results.append(
            result_summary
        )


        task_results[
            task
        ].append(
            result_summary
        )


        # ---------------------------------------------------------------------
        # DISPLAY FINAL TEST RESULT
        # ---------------------------------------------------------------------

        print()

        print(
            "TEST RESULTS"
        )


        print(
            "------------"
        )


        print(
            "Test loss:",
            round(
                evaluation_results[
                    "loss"
                ],
                6
            )
        )


        if task == "classification":

            print(
                "Accuracy:",
                round(
                    evaluation_results[
                        "accuracy"
                    ],
                    6
                )
            )


            print(
                "Balanced accuracy:",
                round(
                    evaluation_results[
                        "balanced_accuracy"
                    ],
                    6
                )
            )


            print(
                "Sensitivity:",
                round(
                    evaluation_results[
                        "sensitivity"
                    ],
                    6
                )
            )


            print(
                "Specificity:",
                round(
                    evaluation_results[
                        "specificity"
                    ],
                    6
                )
            )


        if task == "regression":

            print(
                "RMSE:",
                round(
                    evaluation_results[
                        "rmse"
                    ],
                    6
                )
            )


            print(
                "MAE:",
                round(
                    evaluation_results[
                        "mae"
                    ],
                    6
                )
            )


            print(
                "R-squared:",
                round(
                    evaluation_results[
                        "r2"
                    ],
                    6
                )
            )


        print()

        print(
            "Results saved to:",
            experiment_directory
        )


        # ---------------------------------------------------------------------
        # CLEAN GPU MEMORY BEFORE NEXT MODEL
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


    # =========================================================================
    # 15. SAVE TASK-SPECIFIC RESULTS
    # =========================================================================

    task_results_data = pd.DataFrame(

        task_results[
            task
        ]
    )


    task_results_path = (

        task_result_directory

        /

        "model_results.csv"
    )


    task_results_data.to_csv(

        task_results_path,

        index=False
    )


    print()

    print(
        f"{task.capitalize()} summary saved to:"
    )

    print(
        task_results_path
    )


# =============================================================================
# 16. SAVE COMBINED MODEL RESULTS
# =============================================================================

combined_results = pd.DataFrame(
    all_results
)


combined_results_path = (

    RESULTS_DIRECTORY

    /

    "longgcn_model_results.csv"
)


combined_results.to_csv(

    combined_results_path,

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
    "ALL LONGGCN MODEL FITS COMPLETE"
)

print(
    "=" * 80
)


print()

print(
    "Total models fit:",
    len(
        combined_results
    )
)


print()

print(
    "Combined results saved to:"
)

print(
    combined_results_path
)


print()

print(
    "CLASSIFICATION RESULTS"
)

print(
    "----------------------"
)


classification_results = (
    combined_results[
        combined_results[
            "task"
        ]
        ==
        "classification"
    ]
)


classification_display_columns = [

    "condition",

    "number_of_groups",

    "total_parameters",

    "best_epoch",

    "best_validation_loss",

    "balanced_accuracy",

    "sensitivity",

    "specificity"
]


print(

    classification_results[
        classification_display_columns
    ]
    .to_string(
        index=False
    )
)


print()

print(
    "REGRESSION RESULTS"
)

print(
    "------------------"
)


regression_results = (
    combined_results[
        combined_results[
            "task"
        ]
        ==
        "regression"
    ]
)


regression_display_columns = [

    "condition",

    "number_of_groups",

    "total_parameters",

    "best_epoch",

    "best_validation_loss",

    "rmse",

    "mae",

    "r2"
]


print(

    regression_results[
        regression_display_columns
    ]
    .to_string(
        index=False
    )
)


print()