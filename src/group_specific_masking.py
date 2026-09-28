"""
Group-Specific Measurement Masking
==================================

This module creates structured missing-data patterns in which different
measurement groups are observed at different longitudinal observation times.

The initial masking scheme divides each patient's observation times into two
complementary sets:

    Group 1 measurements: x1, x2, x3
    Group 2 measurements: x4, x5

At observation times assigned to Group 1, measurements x4 and x5 are masked.

At observation times assigned to Group 2, measurements x1, x2, and x3 are
masked.

This creates an extreme structured missing-data setting in which the two
measurement groups are never observed at the same time. The resulting data
can be used to evaluate graph neural network architectures that explicitly
model group-specific longitudinal measurement patterns.

Additional random missingness can later be applied to the masked data using
the separate missing-data generation module.
"""


import numpy as np
import pandas as pd


# =============================================================================
# APPLY GROUP-SPECIFIC MEASUREMENT MASKING
# =============================================================================

def apply_group_specific_masking(
    longitudinal_data,
    group_one_measurements=None,
    group_two_measurements=None,
    group_one_proportion=0.50,
    seed=300
):
    """
    Assign each observation time to one of two measurement groups and mask
    measurements belonging to the opposite group.

    For each patient, observation times are randomly partitioned into two
    complementary sets.

    At Group 1 times:

        x1, x2, x3 are observed
        x4, x5 are masked

    At Group 2 times:

        x1, x2, x3 are masked
        x4, x5 are observed

    By default, approximately half of each patient's visits are assigned to
    each group. Every patient is forced to have at least one observation time
    in each group.

    Parameters
    ----------
    longitudinal_data : pandas.DataFrame
        Complete longitudinal data containing one row per patient visit.

    group_one_measurements : list of str or None
        Measurements belonging to the first measurement group.
        If None, ["x1", "x2", "x3"] is used.

    group_two_measurements : list of str or None
        Measurements belonging to the second measurement group.
        If None, ["x4", "x5"] is used.

    group_one_proportion : float
        Approximate proportion of each patient's observation times assigned
        to Group 1.

    seed : int
        Random seed used to assign observation times to measurement groups.

    Returns
    -------
    masked_longitudinal_data : pandas.DataFrame
        Copy of the longitudinal data after applying group-specific masking.

    group_assignments : pandas.DataFrame
        Patient, visit, and time identifiers together with the assigned
        measurement group for each observation time.

    group_mask : pandas.DataFrame
        Boolean indicators identifying which measurements were removed by
        the group-specific masking process.
    """

    # -------------------------------------------------------------------------
    # 1. INITIALIZE RANDOM NUMBER GENERATOR
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )


    # -------------------------------------------------------------------------
    # 2. DEFINE MEASUREMENT GROUPS
    # -------------------------------------------------------------------------

    if group_one_measurements is None:

        group_one_measurements = [
            "x1",
            "x2",
            "x3"
        ]


    if group_two_measurements is None:

        group_two_measurements = [
            "x4",
            "x5"
        ]


    all_measurements = (
        group_one_measurements
        + group_two_measurements
    )


    # -------------------------------------------------------------------------
    # 3. CREATE OUTPUT DATA SET
    # -------------------------------------------------------------------------

    masked_longitudinal_data = (
        longitudinal_data.copy()
    )


    # -------------------------------------------------------------------------
    # 4. CREATE MEASUREMENT-GROUP ASSIGNMENT TABLE
    # -------------------------------------------------------------------------

    group_assignments = longitudinal_data[
        [
            "patient_id",
            "visit_number",
            "time"
        ]
    ].copy()


    group_assignments[
        "measurement_group"
    ] = None


    # -------------------------------------------------------------------------
    # 5. CREATE GROUP-SPECIFIC MASK TABLE
    # -------------------------------------------------------------------------

    group_mask = longitudinal_data[
        [
            "patient_id",
            "visit_number",
            "time"
        ]
    ].copy()


    for measurement_name in all_measurements:

        group_mask[
            f"{measurement_name}_masked"
        ] = False


    # -------------------------------------------------------------------------
    # 6. PARTITION EACH PATIENT'S OBSERVATION TIMES
    # -------------------------------------------------------------------------

    for patient_id, patient_visits in longitudinal_data.groupby(
        "patient_id",
        sort=False
    ):

        patient_rows = (
            patient_visits.index.to_numpy()
        )

        number_of_visits = len(
            patient_rows
        )


        # Determine approximately how many visits should belong to Group 1.

        number_group_one = int(
            round(
                group_one_proportion
                * number_of_visits
            )
        )


        # Ensure that both groups occur at least once for every patient.

        number_group_one = max(
            1,
            number_group_one
        )

        number_group_one = min(
            number_of_visits - 1,
            number_group_one
        )


        # Randomly choose the observation times belonging to Group 1.

        group_one_rows = rng.choice(
            patient_rows,
            size=number_group_one,
            replace=False
        )


        # All remaining observation times belong to Group 2.

        group_two_rows = np.setdiff1d(
            patient_rows,
            group_one_rows
        )


        # ---------------------------------------------------------------------
        # 7. RECORD GROUP ASSIGNMENTS
        # ---------------------------------------------------------------------

        group_assignments.loc[
            group_one_rows,
            "measurement_group"
        ] = "group_1"


        group_assignments.loc[
            group_two_rows,
            "measurement_group"
        ] = "group_2"


        # ---------------------------------------------------------------------
        # 8. MASK GROUP 2 MEASUREMENTS AT GROUP 1 TIMES
        # ---------------------------------------------------------------------

        masked_longitudinal_data.loc[
            group_one_rows,
            group_two_measurements
        ] = np.nan


        for measurement_name in group_two_measurements:

            group_mask.loc[
                group_one_rows,
                f"{measurement_name}_masked"
            ] = True


        # ---------------------------------------------------------------------
        # 9. MASK GROUP 1 MEASUREMENTS AT GROUP 2 TIMES
        # ---------------------------------------------------------------------

        masked_longitudinal_data.loc[
            group_two_rows,
            group_one_measurements
        ] = np.nan


        for measurement_name in group_one_measurements:

            group_mask.loc[
                group_two_rows,
                f"{measurement_name}_masked"
            ] = True


    # -------------------------------------------------------------------------
    # 10. RETURN MASKED DATA AND MASKING INFORMATION
    # -------------------------------------------------------------------------

    return (
        masked_longitudinal_data,
        group_assignments,
        group_mask
    )