"""
Model Evaluation Utilities
==========================

This module provides evaluation utilities for LongGCN prediction models.

The functions operate on DataLoader objects containing LongGCNBatch objects
created by the installed longgcn package.

Each batch contains:

    batch.y
        Patient-level outcomes.

    batch.num_patients
        Number of patients in the mini-batch.

and can be transferred to the selected computing device using:

    batch = batch.to(device)

The LongGCN prediction model receives the complete batch:

    predictions = model(batch)

Two evaluation routines are provided:

    evaluate_classification_model()
        Evaluates binary classification predictions.

    evaluate_regression_model()
        Evaluates continuous regression predictions.

The model itself always returns raw scalar predictions.

For classification, raw logits are converted to probabilities with the
sigmoid function only during evaluation.

For regression, raw predictions are used directly.
"""


import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =============================================================================
# EVALUATE BINARY CLASSIFICATION MODEL
# =============================================================================

def evaluate_classification_model(
    model,
    data_loader,
    device,
    probability_threshold=0.5
):
    """
    Evaluate a LongGCN binary classification model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LongGCN prediction model.

    data_loader : torch.utils.data.DataLoader
        DataLoader returning LongGCNBatch objects.

    device : torch.device
        CPU or GPU computing device.

    probability_threshold : float
        Probability threshold used to convert predicted probabilities into
        binary predicted classes.

        Defaults to 0.5.

    Returns
    -------
    evaluation_results : dict
        Dictionary containing:

            loss
                Mean BCEWithLogitsLoss across patients.

            accuracy
                Classification accuracy.

            balanced_accuracy
                Mean of sensitivity and specificity.

            sensitivity
                True-positive rate.

            specificity
                True-negative rate.

            true_positives
                Number of correctly classified positive patients.

            true_negatives
                Number of correctly classified negative patients.

            false_positives
                Number of negative patients classified as positive.

            false_negatives
                Number of positive patients classified as negative.

            true_outcomes
                Observed binary outcomes.

            predicted_logits
                Raw model logits.

            predicted_probabilities
                Sigmoid-transformed probabilities.

            predicted_classes
                Thresholded binary predictions.
    """

    # -------------------------------------------------------------------------
    # 1. ENTER EVALUATION MODE
    # -------------------------------------------------------------------------

    model.eval()


    # -------------------------------------------------------------------------
    # 2. CREATE CLASSIFICATION LOSS
    # -------------------------------------------------------------------------

    loss_function = nn.BCEWithLogitsLoss()


    # -------------------------------------------------------------------------
    # 3. INITIALIZE STORAGE
    # -------------------------------------------------------------------------

    total_loss = 0.0

    total_patients = 0


    true_outcomes = []

    predicted_logits = []

    predicted_probabilities = []

    predicted_classes = []


    # -------------------------------------------------------------------------
    # 4. PROCESS MINI-BATCHES
    # -------------------------------------------------------------------------

    with torch.no_grad():

        for batch in data_loader:

            # -----------------------------------------------------------------
            # MOVE COMPLETE LONGGCN BATCH TO DEVICE
            # -----------------------------------------------------------------

            batch = batch.to(
                device
            )


            # -----------------------------------------------------------------
            # GENERATE RAW LOGITS
            # -----------------------------------------------------------------

            logits = model(
                batch
            )


            # -----------------------------------------------------------------
            # CALCULATE MINI-BATCH LOSS
            # -----------------------------------------------------------------

            loss = loss_function(
                logits,
                batch.y
            )


            total_loss += (
                loss.item()
                *
                batch.num_patients
            )


            total_patients += (
                batch.num_patients
            )


            # -----------------------------------------------------------------
            # CONVERT LOGITS TO PROBABILITIES
            # -----------------------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )


            # -----------------------------------------------------------------
            # CONVERT PROBABILITIES TO CLASSES
            # -----------------------------------------------------------------

            classes = (
                probabilities
                >=
                probability_threshold
            ).to(
                dtype=torch.int64
            )


            # -----------------------------------------------------------------
            # SAVE RESULTS ON CPU
            # -----------------------------------------------------------------

            true_outcomes.extend(
                batch.y
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


            predicted_logits.extend(
                logits
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


            predicted_probabilities.extend(
                probabilities
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


            predicted_classes.extend(
                classes
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


    # -------------------------------------------------------------------------
    # 5. CONVERT OUTCOMES TO NUMPY ARRAYS
    # -------------------------------------------------------------------------

    true_outcomes = np.asarray(
        true_outcomes,
        dtype=int
    )


    predicted_logits = np.asarray(
        predicted_logits,
        dtype=float
    )


    predicted_probabilities = np.asarray(
        predicted_probabilities,
        dtype=float
    )


    predicted_classes = np.asarray(
        predicted_classes,
        dtype=int
    )


    # -------------------------------------------------------------------------
    # 6. CALCULATE OVERALL LOSS
    # -------------------------------------------------------------------------

    classification_loss = (
        total_loss
        /
        total_patients
    )


    # -------------------------------------------------------------------------
    # 7. CALCULATE CLASSIFICATION METRICS
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        true_outcomes,
        predicted_classes
    )


    balanced_accuracy = balanced_accuracy_score(
        true_outcomes,
        predicted_classes
    )


    # -------------------------------------------------------------------------
    # 8. CALCULATE CONFUSION MATRIX
    # -------------------------------------------------------------------------

    # Explicit labels ensure that the returned confusion matrix always has
    # shape 2 x 2, even if one class happens to be absent from predictions.

    confusion = confusion_matrix(
        true_outcomes,
        predicted_classes,
        labels=[
            0,
            1
        ]
    )


    true_negatives = int(
        confusion[
            0,
            0
        ]
    )


    false_positives = int(
        confusion[
            0,
            1
        ]
    )


    false_negatives = int(
        confusion[
            1,
            0
        ]
    )


    true_positives = int(
        confusion[
            1,
            1
        ]
    )


    # -------------------------------------------------------------------------
    # 9. CALCULATE SENSITIVITY
    # -------------------------------------------------------------------------

    positive_denominator = (
        true_positives
        +
        false_negatives
    )


    if positive_denominator > 0:

        sensitivity = (
            true_positives
            /
            positive_denominator
        )

    else:

        sensitivity = np.nan


    # -------------------------------------------------------------------------
    # 10. CALCULATE SPECIFICITY
    # -------------------------------------------------------------------------

    negative_denominator = (
        true_negatives
        +
        false_positives
    )


    if negative_denominator > 0:

        specificity = (
            true_negatives
            /
            negative_denominator
        )

    else:

        specificity = np.nan


    # -------------------------------------------------------------------------
    # 11. RETURN EVALUATION RESULTS
    # -------------------------------------------------------------------------

    evaluation_results = {

        "loss":
            classification_loss,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "true_positives":
            true_positives,

        "true_negatives":
            true_negatives,

        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,

        "true_outcomes":
            true_outcomes,

        "predicted_logits":
            predicted_logits,

        "predicted_probabilities":
            predicted_probabilities,

        "predicted_classes":
            predicted_classes
    }


    return evaluation_results


# =============================================================================
# EVALUATE REGRESSION MODEL
# =============================================================================

def evaluate_regression_model(
    model,
    data_loader,
    device
):
    """
    Evaluate a LongGCN continuous regression model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LongGCN prediction model.

    data_loader : torch.utils.data.DataLoader
        DataLoader returning LongGCNBatch objects.

    device : torch.device
        CPU or GPU computing device.

    Returns
    -------
    evaluation_results : dict
        Dictionary containing:

            loss
                Mean squared error calculated during PyTorch evaluation.

            mse
                Mean squared error.

            rmse
                Root mean squared error.

            mae
                Mean absolute error.

            r2
                Coefficient of determination.

            true_outcomes
                Observed continuous outcomes.

            predicted_outcomes
                Predicted continuous outcomes.
    """

    # -------------------------------------------------------------------------
    # 1. ENTER EVALUATION MODE
    # -------------------------------------------------------------------------

    model.eval()


    # -------------------------------------------------------------------------
    # 2. CREATE REGRESSION LOSS
    # -------------------------------------------------------------------------

    loss_function = nn.MSELoss()


    # -------------------------------------------------------------------------
    # 3. INITIALIZE STORAGE
    # -------------------------------------------------------------------------

    total_loss = 0.0

    total_patients = 0


    true_outcomes = []

    predicted_outcomes = []


    # -------------------------------------------------------------------------
    # 4. PROCESS MINI-BATCHES
    # -------------------------------------------------------------------------

    with torch.no_grad():

        for batch in data_loader:

            # -----------------------------------------------------------------
            # MOVE COMPLETE LONGGCN BATCH TO DEVICE
            # -----------------------------------------------------------------

            batch = batch.to(
                device
            )


            # -----------------------------------------------------------------
            # GENERATE CONTINUOUS PREDICTIONS
            # -----------------------------------------------------------------

            predictions = model(
                batch
            )


            # -----------------------------------------------------------------
            # CALCULATE MINI-BATCH LOSS
            # -----------------------------------------------------------------

            loss = loss_function(
                predictions,
                batch.y
            )


            total_loss += (
                loss.item()
                *
                batch.num_patients
            )


            total_patients += (
                batch.num_patients
            )


            # -----------------------------------------------------------------
            # SAVE RESULTS ON CPU
            # -----------------------------------------------------------------

            true_outcomes.extend(
                batch.y
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


            predicted_outcomes.extend(
                predictions
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


    # -------------------------------------------------------------------------
    # 5. CONVERT RESULTS TO NUMPY ARRAYS
    # -------------------------------------------------------------------------

    true_outcomes = np.asarray(
        true_outcomes,
        dtype=float
    )


    predicted_outcomes = np.asarray(
        predicted_outcomes,
        dtype=float
    )


    # -------------------------------------------------------------------------
    # 6. CALCULATE OVERALL PYTORCH LOSS
    # -------------------------------------------------------------------------

    regression_loss = (
        total_loss
        /
        total_patients
    )


    # -------------------------------------------------------------------------
    # 7. CALCULATE REGRESSION METRICS
    # -------------------------------------------------------------------------

    mse = mean_squared_error(
        true_outcomes,
        predicted_outcomes
    )


    rmse = np.sqrt(
        mse
    )


    mae = mean_absolute_error(
        true_outcomes,
        predicted_outcomes
    )


    r2 = r2_score(
        true_outcomes,
        predicted_outcomes
    )


    # -------------------------------------------------------------------------
    # 8. RETURN EVALUATION RESULTS
    # -------------------------------------------------------------------------

    evaluation_results = {

        "loss":
            regression_loss,

        "mse":
            mse,

        "rmse":
            rmse,

        "mae":
            mae,

        "r2":
            r2,

        "true_outcomes":
            true_outcomes,

        "predicted_outcomes":
            predicted_outcomes
    }


    return evaluation_results