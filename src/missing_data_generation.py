"""
Random Missing-Data Generation
==============================

This module introduces additional measurement-level missingness into a
longitudinal data set.

The missingness mechanism implemented here is Missing Completely At Random
(MCAR). Each measurement that is currently observed is independently selected
to become missing with a specified probability.

The input data may either be:

1. the complete simulated longitudinal data, or
2. a data set that already contains structured missingness, such as the
   group-specific measurement masking created in group_specific_masking.py.

Existing missing values are preserved and are not reconsidered by the MCAR
mechanism.

By default, at least one currently observed measurement is retained at every
visit so that an observation time is not completely removed from the
longitudinal record.
"""


import numpy as np
import pandas as pd


# =============================================================================
# APPLY MEASUREMENT-LEVEL MCAR MISSINGNESS
# =============================================================================

def apply_mcar_missingness(
    longitudinal_data,
    missing_rate=0.30,
    measurement_columns=None,
    preserve_at_least_one_measurement=True,
    seed=200
):
    """
    Apply measurement-level MCAR missingness to longitudinal data.

    Each currently observed longitudinal measurement is independently set
    to missing with probability equal to `missing_rate`.

    Measurements that are already missing remain missing and are not counted
    as newly generated MCAR missingness.

    When `preserve_at_least_one_measurement=True`, at least one currently
    observed measurement is retained at every visit.

    Parameters
    ----------
    longitudinal_data : pandas.DataFrame
        Longitudinal data containing one row per patient visit. Measurement
        columns may already contain missing values.

    missing_rate : float
        Probability that each currently observed measurement is made missing.

    measurement_columns : list of str or None
        Names of the longitudinal measurement columns. If None, the default
        measurements x1 through x5 are used.

    preserve_at_least_one_measurement : bool
        If True, at least one currently observed measurement remains observed
        at each visit after MCAR missingness is applied.

    seed : int
        Random seed used to generate the missingness pattern.

    Returns
    -------
    missing_longitudinal_data : pandas.DataFrame
        Copy of the input longitudinal data after additional MCAR missingness
        has been introduced.

    mcar_mask : pandas.DataFrame
        Boolean indicators identifying measurements that were newly removed
        by this MCAR mechanism.
    """

    # -------------------------------------------------------------------------
    # 1. INITIALIZE RANDOM NUMBER GENERATOR
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )


    # -------------------------------------------------------------------------
    # 2. DEFINE LONGITUDINAL MEASUREMENT COLUMNS
    # -------------------------------------------------------------------------

    if measurement_columns is None:

        measurement_columns = [
            "x1",
            "x2",
            "x3",
            "x4",
            "x5"
        ]


    number_of_visits = len(
        longitudinal_data
    )

    number_of_measurements = len(
        measurement_columns
    )


    # -------------------------------------------------------------------------
    # 3. IDENTIFY CURRENTLY OBSERVED MEASUREMENTS
    # -------------------------------------------------------------------------

    # True means the measurement is currently observed.
    #
    # This is important because the input data may already contain structured
    # missingness from the group-specific masking procedure.

    currently_observed = (
        longitudinal_data[
            measurement_columns
        ]
        .notna()
        .to_numpy()
    )


    # -------------------------------------------------------------------------
    # 4. GENERATE RANDOM MCAR MISSINGNESS
    # -------------------------------------------------------------------------

    # Each measurement position is randomly selected for removal with
    # probability equal to missing_rate.

    random_missingness = (
        rng.random(
            size=(
                number_of_visits,
                number_of_measurements
            )
        )
        < missing_rate
    )


    # Only currently observed measurements are eligible to become newly
    # missing.

    mcar_mask_array = (
        random_missingness
        & currently_observed
    )


    # -------------------------------------------------------------------------
    # 5. PRESERVE AT LEAST ONE OBSERVED MEASUREMENT PER VISIT
    # -------------------------------------------------------------------------

    if preserve_at_least_one_measurement:

        for row_index in range(
            number_of_visits
        ):

            observed_measurement_indices = np.where(
                currently_observed[
                    row_index
                ]
            )[0]


            # If no measurements are currently observed at this visit,
            # there is nothing for the MCAR procedure to preserve.

            if len(
                observed_measurement_indices
            ) == 0:

                continue


            # Check whether every currently observed measurement at this visit
            # was selected for removal.

            all_current_measurements_removed = (
                mcar_mask_array[
                    row_index,
                    observed_measurement_indices
                ]
                .all()
            )


            # If so, randomly choose one of those measurements and retain it.

            if all_current_measurements_removed:

                measurement_to_retain = rng.choice(
                    observed_measurement_indices
                )


                mcar_mask_array[
                    row_index,
                    measurement_to_retain
                ] = False


    # -------------------------------------------------------------------------
    # 6. APPLY THE MCAR MASK
    # -------------------------------------------------------------------------

    missing_longitudinal_data = (
        longitudinal_data.copy()
    )


    missing_longitudinal_data.loc[
        :,
        measurement_columns
    ] = (
        missing_longitudinal_data[
            measurement_columns
        ]
        .mask(
            mcar_mask_array
        )
    )


    # -------------------------------------------------------------------------
    # 7. CREATE MCAR MASK DATA FRAME
    # -------------------------------------------------------------------------

    mcar_mask = longitudinal_data[
        [
            "patient_id",
            "visit_number",
            "time"
        ]
    ].copy()


    for measurement_index, measurement_name in enumerate(
        measurement_columns
    ):

        mcar_mask[
            f"{measurement_name}_missing"
        ] = mcar_mask_array[
            :,
            measurement_index
        ]


    # -------------------------------------------------------------------------
    # 8. RETURN DATA WITH MCAR MISSINGNESS
    # -------------------------------------------------------------------------

    return (
        missing_longitudinal_data,
        mcar_mask
    )