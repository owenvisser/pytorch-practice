"""
Deep ReLU Prediction Models
===========================

This module defines the non-graph deep ReLU prediction architecture used as
a comparator to LongGCN.

The purpose of this model is to compare:

    LongGCN
        temporal / graph message passing

versus

    Deep ReLU
        ordinary feed-forward transformations

while keeping the neural-network depth, hidden dimensions, pooling strategy,
and total number of trainable parameters matched as closely as possible.


Current matched architecture
----------------------------

For the primary simulation settings:

    number_of_measurements = 5
    latent_dimension = 8
    hidden_dimension = 8
    number_of_hidden_layers = 4
    pooling = "max"

the Deep ReLU architecture is:

    5 -> 8
        initial linear transformation

    8 -> 8
        ReLU

    8 -> 8
        ReLU

    8 -> 8
        ReLU

    8 -> 8
        ReLU

    max pooling across observation times

    8 -> 1
        patient-level prediction


Parameter count
---------------

Initial layer:

    5 * 8 + 8 = 48

Four hidden layers:

    4 * (8 * 8 + 8) = 288

Output layer:

    8 * 1 + 1 = 9

Total:

    48 + 288 + 9 = 345

This exactly matches the current one-group LongGCN model.


Important distinction from LongGCN
----------------------------------

The model receives the same padded LongGCNBatch object for convenience, but
it uses only:

    batch.X
    batch.time_mask

The following graph objects are intentionally ignored:

    batch.T
    batch.A
    batch.P

Therefore there is:

    no graph convolution,
    no temporal edge weighting,
    no group-specific message passing,
    no communication between visits.

Each visit is transformed independently through the same feed-forward
network. The resulting visit-level representations are then pooled into a
single patient-level representation.

This gives a clean comparison between:

    deep nonlinear feature transformation alone

and

    deep nonlinear feature transformation + graph message passing.
"""


import torch
import torch.nn as nn


# =============================================================================
# DEEP RELU PATIENT-LEVEL PREDICTION MODEL
# =============================================================================

class DeepReLUPredictionModel(
    nn.Module
):
    """
    Ordinary deep ReLU model for patient-level prediction.

    Parameters
    ----------
    number_of_measurements : int
        Number of longitudinal measurements in each complete visit.

    latent_dimension : int, default=8
        Dimension after the initial linear transformation.

    hidden_dimension : int, default=8
        Dimension used by the hidden feed-forward layers.

    number_of_hidden_layers : int, default=4
        Number of ordinary hidden Linear + ReLU transformations.

        The current value of four is chosen to match the four
        message-passing transformations in the primary LongGCN model.

    pooling : str, default="max"
        Patient-level pooling operation.

        Supported options:

            "mean"
            "sum"
            "max"

        The primary comparison uses:

            "max"

        because the current LongGCN simulation also uses max pooling.
    """

    def __init__(
        self,
        number_of_measurements,
        latent_dimension=8,
        hidden_dimension=8,
        number_of_hidden_layers=4,
        pooling="max"
    ):

        super().__init__()


        # ---------------------------------------------------------------------
        # 1. VALIDATE SETTINGS
        # ---------------------------------------------------------------------

        if number_of_measurements <= 0:

            raise ValueError(
                "number_of_measurements must be positive."
            )


        if latent_dimension <= 0:

            raise ValueError(
                "latent_dimension must be positive."
            )


        if hidden_dimension <= 0:

            raise ValueError(
                "hidden_dimension must be positive."
            )


        if number_of_hidden_layers <= 0:

            raise ValueError(
                "number_of_hidden_layers must be positive."
            )


        if pooling not in {

            "mean",
            "sum",
            "max"

        }:

            raise ValueError(
                "pooling must be one of: "
                "'mean', 'sum', or 'max'."
            )


        self.number_of_measurements = (
            number_of_measurements
        )


        self.latent_dimension = (
            latent_dimension
        )


        self.hidden_dimension = (
            hidden_dimension
        )


        self.number_of_hidden_layers = (
            number_of_hidden_layers
        )


        self.pooling = pooling


        # ---------------------------------------------------------------------
        # 2. INITIAL LINEAR TRANSFORMATION
        # ---------------------------------------------------------------------

        # This corresponds dimensionally to the InitialLatentTransform used
        # by LongGCN.
        #
        # Importantly, no ReLU is applied immediately after this layer.
        #
        # This mirrors the current LongGCN architecture, in which the initial
        # latent transformation is followed directly by the first graph layer.

        self.initial_transform = nn.Linear(

            number_of_measurements,

            latent_dimension
        )


        # ---------------------------------------------------------------------
        # 3. HIDDEN FEED-FORWARD LAYERS
        # ---------------------------------------------------------------------

        self.hidden_layers = nn.ModuleList()


        current_dimension = (
            latent_dimension
        )


        for _ in range(
            number_of_hidden_layers
        ):

            hidden_layer = nn.Linear(

                current_dimension,

                hidden_dimension
            )


            self.hidden_layers.append(
                hidden_layer
            )


            current_dimension = (
                hidden_dimension
            )


        # ---------------------------------------------------------------------
        # 4. ACTIVATION
        # ---------------------------------------------------------------------

        self.activation = nn.ReLU()


        # ---------------------------------------------------------------------
        # 5. OUTPUT LAYER
        # ---------------------------------------------------------------------

        self.output_layer = nn.Linear(

            current_dimension,

            1
        )


    # =========================================================================
    # APPLY TIME MASK
    # =========================================================================

    @staticmethod
    def _apply_time_mask(
        H,
        time_mask
    ):
        """
        Set padded time positions to exactly zero.

        Parameters
        ----------
        H : torch.Tensor
            Latent representation:

                B x K x q

        time_mask : torch.Tensor
            Boolean mask:

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
    # PATIENT-LEVEL POOLING
    # =========================================================================

    def _pool(
        self,
        H,
        time_mask
    ):
        """
        Pool visit-level latent representations into patient representations.
        """

        mask = time_mask.unsqueeze(
            dim=-1
        )


        # ---------------------------------------------------------------------
        # SUM POOLING
        # ---------------------------------------------------------------------

        if self.pooling == "sum":

            return (

                H

                *

                mask.to(
                    dtype=H.dtype
                )

            ).sum(
                dim=1
            )


        # ---------------------------------------------------------------------
        # MEAN POOLING
        # ---------------------------------------------------------------------

        if self.pooling == "mean":

            masked_H = (

                H

                *

                mask.to(
                    dtype=H.dtype
                )
            )


            numerator = masked_H.sum(
                dim=1
            )


            denominator = (

                time_mask

                .sum(
                    dim=1,
                    keepdim=True
                )

                .clamp_min(
                    1
                )

                .to(
                    dtype=H.dtype
                )
            )


            return (
                numerator
                /
                denominator
            )


        # ---------------------------------------------------------------------
        # MAX POOLING
        # ---------------------------------------------------------------------

        if self.pooling == "max":

            # Padded positions must not participate in max pooling.
            #
            # They are temporarily replaced by the smallest finite value for
            # the current floating-point dtype.

            minimum_value = torch.finfo(
                H.dtype
            ).min


            masked_H = H.masked_fill(

                ~mask,

                minimum_value
            )


            return masked_H.max(
                dim=1
            ).values


        raise RuntimeError(
            "Unexpected pooling option."
        )


    # =========================================================================
    # ENCODE PATIENTS
    # =========================================================================

    def encode(
        self,
        batch
    ):
        """
        Create one fixed-length representation per patient.

        The graph tensors contained in batch are intentionally ignored.
        """

        # ---------------------------------------------------------------------
        # 1. INITIAL TRANSFORMATION
        # ---------------------------------------------------------------------

        H = self.initial_transform(
            batch.X
        )


        # The layer contains a bias, so padded rows must be explicitly masked.

        H = self._apply_time_mask(

            H,

            batch.time_mask
        )


        # ---------------------------------------------------------------------
        # 2. ORDINARY DEEP RELU TRANSFORMATIONS
        # ---------------------------------------------------------------------

        for hidden_layer in self.hidden_layers:

            H = hidden_layer(
                H
            )


            H = self.activation(
                H
            )


            H = self._apply_time_mask(

                H,

                batch.time_mask
            )


        # ---------------------------------------------------------------------
        # 3. PATIENT-LEVEL POOLING
        # ---------------------------------------------------------------------

        patient_embeddings = self._pool(

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
        Return one raw scalar prediction per patient.
        """

        patient_embeddings = self.encode(
            batch
        )


        predictions = self.output_layer(
            patient_embeddings
        )


        predictions = predictions.squeeze(
            dim=-1
        )


        return predictions