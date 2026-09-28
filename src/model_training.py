"""
Model Training Utilities
========================

This module provides training utilities for the LongGCN simulation study.

The functions operate on DataLoader objects containing LongGCNBatch objects
created by the installed longgcn package.

Each batch contains:

    batch.X
    batch.A
    batch.P
    batch.T
    batch.time_mask
    batch.y
    batch.num_patients

and can be transferred to the selected computing device using:

    batch = batch.to(device)

The prediction model receives the complete LongGCNBatch:

    predictions = model(batch)

The same training routine supports both:

    binary classification
        BCEWithLogitsLoss

    continuous regression
        MSELoss

The learning-rate scheduler monitors validation loss after every epoch.

Model checkpointing is performed only at specified checkpoint intervals.
At each checkpoint epoch, the current model is retained only when its
validation loss is strictly lower than the best checkpoint validation loss
observed previously.

This separates:

    learning-rate adaptation
        evaluated every epoch

from:

    model checkpoint selection
        evaluated every checkpoint_interval epochs
"""


import copy

import torch
import torch.nn as nn


# =============================================================================
# GET TASK-SPECIFIC LOSS FUNCTION
# =============================================================================

def get_loss_function(
    task
):
    """
    Return the appropriate loss function for the prediction task.

    Parameters
    ----------
    task : str
        Prediction task.

        Supported options:

            "classification"
            "regression"

    Returns
    -------
    loss_function : torch.nn.Module
        PyTorch loss function.
    """

    if task == "classification":

        return nn.BCEWithLogitsLoss()


    if task == "regression":

        return nn.MSELoss()


    raise ValueError(
        "task must be either "
        "'classification' or 'regression'."
    )


# =============================================================================
# CALCULATE VALIDATION LOSS
# =============================================================================

def calculate_validation_loss(
    model,
    validation_loader,
    loss_function,
    device
):
    """
    Calculate patient-level validation loss.

    Parameters
    ----------
    model : torch.nn.Module
        LongGCN prediction model.

    validation_loader : torch.utils.data.DataLoader
        DataLoader returning LongGCNBatch objects.

    loss_function : torch.nn.Module
        Loss function used for the current prediction task.

    device : torch.device
        CPU or GPU computing device.

    Returns
    -------
    validation_loss : float
        Mean validation loss across patients.
    """

    # -------------------------------------------------------------------------
    # 1. ENTER EVALUATION MODE
    # -------------------------------------------------------------------------

    model.eval()


    # -------------------------------------------------------------------------
    # 2. INITIALIZE LOSS ACCUMULATORS
    # -------------------------------------------------------------------------

    total_validation_loss = 0.0

    total_validation_patients = 0


    # -------------------------------------------------------------------------
    # 3. PROCESS VALIDATION MINI-BATCHES
    # -------------------------------------------------------------------------

    with torch.no_grad():

        for batch in validation_loader:

            # -----------------------------------------------------------------
            # MOVE COMPLETE LONGGCN BATCH TO DEVICE
            # -----------------------------------------------------------------

            batch = batch.to(
                device
            )


            # -----------------------------------------------------------------
            # GENERATE PATIENT-LEVEL PREDICTIONS
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


            # -----------------------------------------------------------------
            # ACCUMULATE PATIENT-WEIGHTED LOSS
            # -----------------------------------------------------------------

            # PyTorch loss functions return the mean loss within the mini-batch
            # by default.
            #
            # Multiplying by the number of patients allows us to reconstruct
            # the overall patient-level mean loss across the validation set.

            total_validation_loss += (
                loss.item()
                *
                batch.num_patients
            )


            total_validation_patients += (
                batch.num_patients
            )


    # -------------------------------------------------------------------------
    # 4. CALCULATE OVERALL VALIDATION LOSS
    # -------------------------------------------------------------------------

    validation_loss = (
        total_validation_loss
        /
        total_validation_patients
    )


    return validation_loss


# =============================================================================
# TRAIN LONGGCN MODEL
# =============================================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    task,
    device,
    number_of_epochs=1000,
    learning_rate=0.001,
    weight_decay=0.0,
    scheduler_factor=0.5,
    scheduler_patience=20,
    minimum_learning_rate=1e-6,
    early_stopping_patience=100,
    minimum_validation_improvement=1e-4,
    checkpoint_interval=20,
    print_interval=20
):
    """
    Train a LongGCN prediction model.

    Parameters
    ----------
    model : torch.nn.Module
        LongGCN prediction model.

    train_loader : torch.utils.data.DataLoader
        Training DataLoader returning LongGCNBatch objects.

    validation_loader : torch.utils.data.DataLoader
        Validation DataLoader returning LongGCNBatch objects.

    task : str
        Prediction task.

        Supported options:

            "classification"
            "regression"

    device : torch.device
        CPU or GPU computing device.

    number_of_epochs : int
        Number of training epochs.

        Defaults to 1000.

    learning_rate : float
        Initial Adam learning rate.

        Defaults to 0.001.

    weight_decay : float
        Adam weight-decay parameter.

        Defaults to 0.0.

    scheduler_factor : float
        Multiplicative learning-rate reduction applied by
        ReduceLROnPlateau.

        For example:

            0.5

        reduces the current learning rate by half.

    scheduler_patience : int
        Number of validation-loss epochs without improvement before reducing
        the learning rate.

    minimum_learning_rate : float
        Minimum learning rate allowed by the scheduler.

    checkpoint_interval : int
        Number of epochs between model checkpoint evaluations.

        For example:

            checkpoint_interval = 20

        checks the model at epochs:

            20, 40, 60, ...

        The model is retained only when the current checkpoint validation loss
        is strictly lower than the previous best checkpoint validation loss.

    print_interval : int or None
        Number of epochs between printed training summaries.

        Set to None to suppress progress printing.

    Returns
    -------
    training_results : dict
        Dictionary containing:

            model
                Model restored to the best checkpoint.

            training_losses
                Training loss after every epoch.

            validation_losses
                Validation loss after every epoch.

            learning_rates
                Learning rate after every epoch.

            checkpoint_history
                Validation results at checkpoint epochs.

            best_epoch
                Epoch of the retained checkpoint.

            best_validation_loss
                Validation loss at the retained checkpoint.
    """

    # -------------------------------------------------------------------------
    # 1. MOVE MODEL TO COMPUTING DEVICE
    # -------------------------------------------------------------------------

    model = model.to(
        device
    )


    # -------------------------------------------------------------------------
    # 2. CREATE TASK-SPECIFIC LOSS FUNCTION
    # -------------------------------------------------------------------------

    loss_function = get_loss_function(
        task
    )


    # -------------------------------------------------------------------------
    # 3. CREATE OPTIMIZER
    # -------------------------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate,

        weight_decay=weight_decay
    )


    # -------------------------------------------------------------------------
    # 4. CREATE LEARNING-RATE SCHEDULER
    # -------------------------------------------------------------------------

    # The scheduler observes validation loss every epoch.
    #
    # This is independent of model checkpointing.

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=scheduler_factor,

        patience=scheduler_patience,

        threshold=minimum_validation_improvement,

        threshold_mode="abs",

        min_lr=minimum_learning_rate
    )


    # -------------------------------------------------------------------------
    # 5. INITIALIZE TRAINING HISTORY
    # -------------------------------------------------------------------------

    training_losses = []

    validation_losses = []

    learning_rates = []

    checkpoint_history = []


    # -------------------------------------------------------------------------
    # 6. INITIALIZE BEST CHECKPOINT
    # -------------------------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_epoch = None

    best_model_state = None

    # -------------------------------------------------------------------------
    # INITIALIZE EARLY STOPPING
    # -------------------------------------------------------------------------

    early_stopping_reference_loss = float(
        "inf"
    )

    epochs_without_meaningful_improvement = 0

    stopped_early = False

    stop_epoch = number_of_epochs

    # =========================================================================
    # 7. TRAINING LOOP
    # =========================================================================

    for epoch in range(
        1,
        number_of_epochs + 1
    ):

        # ---------------------------------------------------------------------
        # ENTER TRAINING MODE
        # ---------------------------------------------------------------------

        model.train()


        # ---------------------------------------------------------------------
        # INITIALIZE EPOCH LOSS
        # ---------------------------------------------------------------------

        total_training_loss = 0.0

        total_training_patients = 0


        # =====================================================================
        # TRAINING MINI-BATCH LOOP
        # =====================================================================

        for batch in train_loader:

            # -----------------------------------------------------------------
            # MOVE COMPLETE LONGGCN BATCH TO DEVICE
            # -----------------------------------------------------------------

            batch = batch.to(
                device
            )


            # -----------------------------------------------------------------
            # RESET GRADIENTS
            # -----------------------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )


            # -----------------------------------------------------------------
            # FORWARD PASS
            # -----------------------------------------------------------------

            predictions = model(
                batch
            )


            # -----------------------------------------------------------------
            # CALCULATE LOSS
            # -----------------------------------------------------------------

            loss = loss_function(
                predictions,
                batch.y
            )


            # -----------------------------------------------------------------
            # BACKPROPAGATION
            # -----------------------------------------------------------------

            loss.backward()


            # -----------------------------------------------------------------
            # UPDATE MODEL PARAMETERS
            # -----------------------------------------------------------------

            optimizer.step()


            # -----------------------------------------------------------------
            # ACCUMULATE PATIENT-WEIGHTED TRAINING LOSS
            # -----------------------------------------------------------------

            total_training_loss += (
                loss.item()
                *
                batch.num_patients
            )


            total_training_patients += (
                batch.num_patients
            )


        # ---------------------------------------------------------------------
        # CALCULATE EPOCH TRAINING LOSS
        # ---------------------------------------------------------------------

        training_loss = (
            total_training_loss
            /
            total_training_patients
        )


        # ---------------------------------------------------------------------
        # CALCULATE VALIDATION LOSS
        # ---------------------------------------------------------------------

        validation_loss = calculate_validation_loss(

            model=model,

            validation_loader=validation_loader,

            loss_function=loss_function,

            device=device
        )


        # ---------------------------------------------------------------------
        # UPDATE LEARNING-RATE SCHEDULER
        # ---------------------------------------------------------------------

        # The scheduler evaluates validation performance after every epoch,
        # rather than only at checkpoint epochs.

        scheduler.step(
            validation_loss
        )


        # ---------------------------------------------------------------------
        # GET CURRENT LEARNING RATE
        # ---------------------------------------------------------------------

        current_learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )


        # ---------------------------------------------------------------------
        # SAVE EPOCH HISTORY
        # ---------------------------------------------------------------------

        training_losses.append(
            training_loss
        )

        validation_losses.append(
            validation_loss
        )

        learning_rates.append(
            current_learning_rate
        )


        # =====================================================================
        # 8. CHECKPOINT MODEL
        # =====================================================================

        if epoch % checkpoint_interval == 0:

            checkpoint_improved = (
                validation_loss
                <
                best_validation_loss
            )


            checkpoint_history.append(
                {
                    "epoch": epoch,

                    "validation_loss": validation_loss,

                    "learning_rate": current_learning_rate,

                    "improved": checkpoint_improved
                }
            )


            # -----------------------------------------------------------------
            # RETAIN STRICTLY BETTER CHECKPOINT
            # -----------------------------------------------------------------

            if checkpoint_improved:

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = (
                    epoch
                )


                # Deep-copy the model state so subsequent training epochs do
                # not modify the retained checkpoint.

                best_model_state = copy.deepcopy(
                    model.state_dict()
                )




        # =====================================================================
        # EARLY STOPPING
        # =====================================================================

        meaningful_improvement = (

            validation_loss

            <

            (
                early_stopping_reference_loss
                -
                minimum_validation_improvement
            )
        )


        if meaningful_improvement:

            early_stopping_reference_loss = (
                validation_loss
            )

            epochs_without_meaningful_improvement = 0


        else:

            epochs_without_meaningful_improvement += 1


        if (
            epochs_without_meaningful_improvement
            >=
            early_stopping_patience
        ):

            stopped_early = True

            stop_epoch = epoch


            print()

            print(
                "Early stopping triggered at "
                f"epoch {epoch}."
            )

            print(
                "No validation-loss improvement of at least "
                f"{minimum_validation_improvement} "
                f"for {early_stopping_patience} epochs."
            )

            print(
                "Best validation loss:",
                round(
                    best_validation_loss,
                    6
                )
            )

            print(
                "Best epoch:",
                best_epoch
            )

            print()


            break


        # =====================================================================
        # 9. PRINT TRAINING PROGRESS
        # =====================================================================

        if (
            print_interval is not None
            and
            epoch % print_interval == 0
        ):

            print(
                f"Epoch {epoch:4d} | "
                f"Training Loss: {training_loss:.6f} | "
                f"Validation Loss: {validation_loss:.6f} | "
                f"Learning Rate: {current_learning_rate:.8f}"
            )


    # =========================================================================
    # 10. HANDLE TRAINING WITHOUT A CHECKPOINT EPOCH
    # =========================================================================

    # With the intended settings this normally will not occur because the
    # number of training epochs is much larger than checkpoint_interval.
    #
    # This fallback simply preserves the final trained model if no checkpoint
    # epoch occurred.

    if best_model_state is None:

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = (
            number_of_epochs
        )

        best_validation_loss = (
            validation_losses[-1]
        )


    # =========================================================================
    # 11. RESTORE BEST CHECKPOINT
    # =========================================================================

    model.load_state_dict(
        best_model_state
    )


    # =========================================================================
    # 12. RETURN TRAINING RESULTS
    # =========================================================================

    training_results = {

        "model":
            model,

        "training_losses":
            training_losses,

        "validation_losses":
            validation_losses,

        "learning_rates":
            learning_rates,

        "checkpoint_history":
            checkpoint_history,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_validation_loss,

        "epochs_completed":
            len(
                training_losses
            ),

        "stopped_early":
            stopped_early,

        "stop_epoch":
            stop_epoch,

        "early_stopping_patience":
            early_stopping_patience,

        "minimum_validation_improvement":
            minimum_validation_improvement
    }

    return training_results