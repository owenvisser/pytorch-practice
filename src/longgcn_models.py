"""
LongGCN Prediction Models
=========================

This module defines the LongGCN prediction architecture used in the simulation
study.

The low-level longitudinal representation, graph construction, padded
mini-batching, and graph neural-network layers are provided by the installed
longgcn package.

This module is responsible only for assembling those reusable components into
a patient-level prediction model.

The model receives a LongGCNBatch containing:

    X
        B x K_max x M

    A
        B x G x K_max x K_max

    P
        B x G x K_max x K_max

    T
        B x K_max x K_max

    time_mask
        B x K_max

and applies:

    InitialLatentTransform
        ↓
    GroupSpecificGraphLayer(s)
        ↓
    TemporalGraphLayer(s)
        ↓
    Patient-level pooling
        ↓
    Linear prediction layer

The final model produces one raw scalar prediction for each patient:

    B

No task-specific output transformation is included in the model.

For binary classification, the raw values are logits and should be used with:

    BCEWithLogitsLoss

For regression, the same raw values should be used directly with:

    MSELoss

This allows the exact same LongGCN architecture to be compared across
classification and regression tasks.
"""


import torch
import torch.nn as nn

from longgcn.nn import (
    InitialLatentTransform,
    GroupSpecificGraphLayer,
    TemporalGraphLayer,
    MeanPool,
    SumPool,
    MaxPool,
    MeanMaxPool
)


# =============================================================================
# CREATE PATIENT-LEVEL POOLING LAYER
# =============================================================================

def create_pooling_layer(
    pooling
):
    """
    Create a LongGCN patient-level pooling layer.

    Parameters
    ----------
    pooling : str
        Pooling method.

        Supported options:

            "mean"
            "sum"
            "max"
            "mean_max"

    Returns
    -------
    pooling_layer : nn.Module
        LongGCN pooling layer.
    """

    if pooling == "mean":

        return MeanPool()


    if pooling == "sum":

        return SumPool()


    if pooling == "max":

        return MaxPool()


    if pooling == "mean_max":

        return MeanMaxPool()


    raise ValueError(
        "pooling must be one of: "
        "'mean', 'sum', 'max', or 'mean_max'."
    )


# =============================================================================
# LONGGCN PATIENT-LEVEL PREDICTION MODEL
# =============================================================================

class LongGCNPredictionModel(
    nn.Module
):
    """
    LongGCN model for patient-level prediction.

    Parameters
    ----------
    number_of_measurements : int
        Number of possible longitudinal measurement types M.

        For the current simulation:

            M = 5

    group_names : list[str]
        Ordered names of the predefined measurement groups.

        The order must match the group ordering used by
        DesignedMeasurementGroups when constructing the LongGCNTorchDataset.

        For example, the complete-data model may use:

            ["all_measurements"]

        while the group-specific model may use:

            ["group_1", "group_2"]

    latent_dimension : int
        Dimension of the initial latent representation.

        Defaults to 8.

    hidden_dimension : int
        Dimension used by the graph message-passing layers.

        Defaults to 8.

    number_of_group_layers : int
        Number of group-specific graph layers.

        Defaults to 2.

    number_of_temporal_layers : int
        Number of unrestricted temporal graph layers.

        Defaults to 2.

    group_aggregation : str
        Aggregation method used by GroupSpecificGraphLayer.

        Supported options:

            "sum"
            "mean"

    temporal_aggregation : str
        Aggregation method used by TemporalGraphLayer.

        Supported options:

            "sum"
            "mean"

    parameter_sharing : str
        Parameter-sharing structure across designed measurement groups.

        Supported options:

            "group"
                Each group receives its own W_g and b_g.

            "shared"
                All groups use the same W and b.

    pooling : str
        Patient-level pooling method.

        Supported options:

            "mean"
            "sum"
            "max"
            "mean_max"
    """

    def __init__(
        self,
        number_of_measurements,
        group_names,
        latent_dimension=8,
        hidden_dimension=8,
        number_of_group_layers=2,
        number_of_temporal_layers=2,
        group_aggregation="sum",
        temporal_aggregation="sum",
        parameter_sharing="group",
        pooling="max"
    ):

        super().__init__()


        # ---------------------------------------------------------------------
        # 1. SAVE MODEL SETTINGS
        # ---------------------------------------------------------------------

        self.number_of_measurements = (
            number_of_measurements
        )

        self.group_names = list(
            group_names
        )

        self.number_of_groups = len(
            self.group_names
        )

        self.latent_dimension = (
            latent_dimension
        )

        self.hidden_dimension = (
            hidden_dimension
        )

        self.number_of_group_layers = (
            number_of_group_layers
        )

        self.number_of_temporal_layers = (
            number_of_temporal_layers
        )

        self.group_aggregation = (
            group_aggregation
        )

        self.temporal_aggregation = (
            temporal_aggregation
        )

        self.parameter_sharing = (
            parameter_sharing
        )

        self.pooling_method = (
            pooling
        )


        # ---------------------------------------------------------------------
        # 2. INITIAL MEASUREMENT-TO-LATENT TRANSFORMATION
        # ---------------------------------------------------------------------

        # X:
        #
        #     B x K x M
        #
        # becomes:
        #
        #     B x K x latent_dimension

        self.initial_transform = InitialLatentTransform(

            n_measurements=number_of_measurements,

            latent_dim=latent_dimension
        )


        # ---------------------------------------------------------------------
        # 3. GROUP-SPECIFIC MESSAGE-PASSING LAYERS
        # ---------------------------------------------------------------------

        self.group_layers = nn.ModuleList()


        current_dimension = (
            latent_dimension
        )


        for layer_index in range(
            number_of_group_layers
        ):

            group_layer = GroupSpecificGraphLayer(

                group_names=self.group_names,

                input_dim=current_dimension,

                output_dim=hidden_dimension,

                aggregation=group_aggregation,

                parameter_sharing=parameter_sharing
            )


            self.group_layers.append(
                group_layer
            )


            current_dimension = (
                hidden_dimension
            )


        # ---------------------------------------------------------------------
        # 4. UNRESTRICTED TEMPORAL MESSAGE-PASSING LAYERS
        # ---------------------------------------------------------------------

        self.temporal_layers = nn.ModuleList()


        for layer_index in range(
            number_of_temporal_layers
        ):

            temporal_layer = TemporalGraphLayer(

                input_dim=current_dimension,

                output_dim=hidden_dimension,

                aggregation=temporal_aggregation
            )


            self.temporal_layers.append(
                temporal_layer
            )


            current_dimension = (
                hidden_dimension
            )


        # ---------------------------------------------------------------------
        # 5. ACTIVATION FUNCTION
        # ---------------------------------------------------------------------

        # Activation is intentionally defined outside the individual LongGCN
        # layers.
        #
        # This keeps the graph layers themselves purely responsible for the
        # linear/message-passing operation.

        self.activation = nn.ReLU()


        # ---------------------------------------------------------------------
        # 6. PATIENT-LEVEL POOLING
        # ---------------------------------------------------------------------

        self.pooling = create_pooling_layer(
            pooling
        )


        # ---------------------------------------------------------------------
        # 7. DETERMINE POOLED REPRESENTATION DIMENSION
        # ---------------------------------------------------------------------

        # Mean, sum, and max pooling retain the latent dimension.
        #
        # Mean-max pooling concatenates two representations and therefore
        # doubles the dimension.

        if pooling == "mean_max":

            pooled_dimension = (
                2
                *
                current_dimension
            )


        else:

            pooled_dimension = (
                current_dimension
            )


        self.pooled_dimension = (
            pooled_dimension
        )


        # ---------------------------------------------------------------------
        # 8. PATIENT-LEVEL PREDICTION LAYER
        # ---------------------------------------------------------------------

        # One raw scalar is returned for each patient.
        #
        # Classification:
        #
        #     raw scalar = logit
        #
        # Regression:
        #
        #     raw scalar = predicted continuous outcome

        self.output_layer = nn.Linear(
            pooled_dimension,
            1
        )


    # =========================================================================
    # MASK PADDED TIME POSITIONS
    # =========================================================================

    @staticmethod
    def _apply_time_mask(
        H,
        time_mask
    ):
        """
        Set padded latent positions to exactly zero.

        Parameters
        ----------
        H : torch.Tensor
            Batched latent representation:

                B x K x q

        time_mask : torch.Tensor
            Boolean real-time mask:

                B x K

        Returns
        -------
        H : torch.Tensor
            Masked latent representation.
        """

        return (
            H
            *
            time_mask.unsqueeze(
                dim=-1
            ).to(
                dtype=H.dtype
            )
        )


    # =========================================================================
    # ENCODE PATIENTS
    # =========================================================================

    def encode(
        self,
        batch
    ):
        """
        Encode a LongGCNBatch into fixed-length patient representations.

        Parameters
        ----------
        batch : LongGCNBatch
            Padded mini-batch created by collate_longgcn.

        Returns
        -------
        patient_embeddings : torch.Tensor
            Fixed-length patient representations:

                B x pooled_dimension
        """

        # ---------------------------------------------------------------------
        # 1. INITIAL LATENT TRANSFORMATION
        # ---------------------------------------------------------------------

        H = self.initial_transform(
            batch.X
        )


        # The initial transformation includes a bias term. Therefore padded
        # zero rows could become nonzero after the transformation.
        #
        # Explicit masking restores the invariant that padded latent positions
        # are exactly zero.

        H = self._apply_time_mask(
            H,
            batch.time_mask
        )


        # ---------------------------------------------------------------------
        # 2. GROUP-SPECIFIC MESSAGE PASSING
        # ---------------------------------------------------------------------

        for group_layer in self.group_layers:

            H = group_layer(

                H=H,

                A=batch.A,

                P=batch.P,

                time_mask=batch.time_mask
            )


            H = self.activation(
                H
            )


            H = self._apply_time_mask(
                H,
                batch.time_mask
            )


        # ---------------------------------------------------------------------
        # 3. UNRESTRICTED TEMPORAL MESSAGE PASSING
        # ---------------------------------------------------------------------

        for temporal_layer in self.temporal_layers:

            H = temporal_layer(

                H=H,

                T=batch.T,

                time_mask=batch.time_mask
            )


            H = self.activation(
                H
            )


            H = self._apply_time_mask(
                H,
                batch.time_mask
            )


        # ---------------------------------------------------------------------
        # 4. PATIENT-LEVEL POOLING
        # ---------------------------------------------------------------------

        patient_embeddings = self.pooling(

            H,

            batch.time_mask
        )


        return patient_embeddings


    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(
        self,
        batch
    ):
        """
        Generate one raw patient-level prediction per patient.

        Parameters
        ----------
        batch : LongGCNBatch
            Padded LongGCN mini-batch.

        Returns
        -------
        predictions : torch.Tensor
            Raw patient-level predictions with shape:

                B
        """

        # ---------------------------------------------------------------------
        # 1. CREATE PATIENT REPRESENTATIONS
        # ---------------------------------------------------------------------

        patient_embeddings = self.encode(
            batch
        )


        # ---------------------------------------------------------------------
        # 2. GENERATE RAW PATIENT-LEVEL PREDICTIONS
        # ---------------------------------------------------------------------

        predictions = self.output_layer(
            patient_embeddings
        )


        # Remove only the final singleton output dimension:
        #
        #     B x 1
        #
        # becomes:
        #
        #     B
        #
        # Using squeeze(-1), rather than squeeze(), is important because a
        # mini-batch containing only one patient should still produce shape:
        #
        #     [1]
        #
        # rather than a scalar.

        predictions = predictions.squeeze(
            dim=-1
        )

        return predictions