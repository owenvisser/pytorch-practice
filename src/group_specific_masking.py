"""
Group-Specific Measurement Masking
==================================

This module creates structured longitudinal missingness based on predefined
measurement groups.

For the current simulation, the default measurement groups are:

    Group 1
        x1, x2, x3

    Group 2
        x4, x5

Under the standard group-specific masking mechanism, each patient has some
observation times assigned to Group 1 and the remaining observation times
assigned to Group 2.

At a Group 1 observation time:

    x1, x2, x3 are retained
    x4, x5 are masked

At a Group 2 observation time:

    x4, x5 are retained
    x1, x2, x3 are masked

The function also allows selected patients to be assigned exclusively to
Group 1. Such patients never receive any Group 2 measurements and therefore
have:

    x4 = missing
    x5 = missing

at every observation time.

This provides a useful stress test for LongGCN because the corresponding
patient has no observations from the second designed measurement group.
"""


import numpy as np
import pandas as pd


# =============================================================================
# APPLY GROUP-SPECIFIC MASKING
# =============================================================================

def apply_group_specific_masking(
    longitudinal_data,
    group_one_measurements=None,
    group_two_measurements=None,
    group_one_proportion=0.50,
    group_one_only_patient_ids=None,
    seed=300
):
    """
    Apply structured group-specific measurement masking.

    Parameters
    ----------
    longitudinal_data : pandas.DataFrame
        Complete visit-level longitudinal data.

    group_one_measurements : list[str] or None
        Measurements belonging to Group 1.

        Defaults to:

            x1, x2, x3

    group_two_measurements : list[str] or None
        Measurements belonging to Group 2.

        Defaults to:

            x4, x5

    group_one_proportion : float
        Approximate proportion of observation times assigned to Group 1 for
        patients following the standard two-group observation structure.

        Defaults to 0.50.

    group_one_only_patient_ids : iterable or None
        Optional patient IDs that should never receive Group 2 measurements.

        Every observation time for these patients is assigned to Group 1.

        Therefore all Group 2 measurements are missing across the patient's
        complete longitudinal record.

    seed : int
        Random seed used for assigning observation times to groups.

    Returns
    -------
    masked_longitudinal_data : pandas.DataFrame
        Longitudinal data after structural measurement masking.

    group_assignments : pandas.DataFrame
        Patient/time-level group assignment.

    group_mask : pandas.DataFrame
        Boolean indicators showing which measurement cells were structurally
        masked.
    """

    # -------------------------------------------------------------------------
    # 1. DEFAULT MEASUREMENT GROUPS
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


    # -------------------------------------------------------------------------
    # 2. PATIENTS WITH NO GROUP 2 OBSERVATIONS
    # -------------------------------------------------------------------------

    if group_one_only_patient_ids is None:

        group_one_only_patient_ids = set()


    else:

        group_one_only_patient_ids = set(
            group_one_only_patient_ids
        )


    # -------------------------------------------------------------------------
    # 3. RANDOM NUMBER GENERATOR
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )


    # -------------------------------------------------------------------------
    # 4. COPY COMPLETE LONGITUDINAL DATA
    # -------------------------------------------------------------------------

    masked_longitudinal_data = (
        longitudinal_data.copy()
    )


    # -------------------------------------------------------------------------
    # 5. CREATE GROUP-ASSIGNMENT TABLE
    # -------------------------------------------------------------------------

    group_assignments = (
        longitudinal_data[
            [
                "patient_id",
                "visit_number",
                "time"
            ]
        ]
        .copy()
    )


    group_assignments[
        "measurement_group"
    ] = None


    # -------------------------------------------------------------------------
    # 6. CREATE STRUCTURAL MASK TABLE
    # -------------------------------------------------------------------------

    group_mask = (
        longitudinal_data[
            [
                "patient_id",
                "visit_number",
                "time"
            ]
        ]
        .copy()
    )


    all_measurements = (
        group_one_measurements
        +
        group_two_measurements
    )


    for measurement in all_measurements:

        group_mask[
            f"{measurement}_masked"
        ] = False


    # =========================================================================
    # 7. ASSIGN OBSERVATION TIMES WITHIN EACH PATIENT
    # =========================================================================

    for patient_id, patient_data in longitudinal_data.groupby(
        "patient_id",
        sort=False
    ):

        patient_indices = (
            patient_data
            .index
            .to_numpy()
        )


        number_of_times = len(
            patient_indices
        )


        # ---------------------------------------------------------------------
        # PATIENT HAS NO GROUP 2 MEASUREMENTS
        # ---------------------------------------------------------------------

        if patient_id in group_one_only_patient_ids:

            group_one_indices = (
                patient_indices
            )


            group_two_indices = np.array(
                [],
                dtype=patient_indices.dtype
            )


        # ---------------------------------------------------------------------
        # STANDARD TWO-GROUP PATIENT
        # ---------------------------------------------------------------------

        else:

            # Determine approximately how many observation times belong to
            # Group 1.

            number_group_one = int(
                round(
                    group_one_proportion
                    *
                    number_of_times
                )
            )


            # For ordinary patients, retain at least one observation time in
            # each group whenever the patient has at least two times.

            if number_of_times >= 2:

                number_group_one = max(
                    1,
                    min(
                        number_group_one,
                        number_of_times - 1
                    )
                )


            else:

                number_group_one = 1


            # Randomly select Group 1 observation times.

            group_one_indices = rng.choice(

                patient_indices,

                size=number_group_one,

                replace=False
            )


            # All remaining observation times belong to Group 2.

            group_two_indices = np.setdiff1d(

                patient_indices,

                group_one_indices
            )


        # =====================================================================
        # 8. APPLY GROUP 1 MASKING
        # =====================================================================

        group_assignments.loc[
            group_one_indices,
            "measurement_group"
        ] = "group_1"


        # At Group 1 times, Group 2 measurements are structurally missing.

        masked_longitudinal_data.loc[
            group_one_indices,
            group_two_measurements
        ] = np.nan


        for measurement in group_two_measurements:

            group_mask.loc[
                group_one_indices,
                f"{measurement}_masked"
            ] = True


        # =====================================================================
        # 9. APPLY GROUP 2 MASKING
        # =====================================================================

        if len(
            group_two_indices
        ) > 0:

            group_assignments.loc[
                group_two_indices,
                "measurement_group"
            ] = "group_2"


            # At Group 2 times, Group 1 measurements are structurally missing.

            masked_longitudinal_data.loc[
                group_two_indices,
                group_one_measurements
            ] = np.nan


            for measurement in group_one_measurements:

                group_mask.loc[
                    group_two_indices,
                    f"{measurement}_masked"
                ] = True


    # -------------------------------------------------------------------------
    # 10. RETURN STRUCTURED-MISSINGNESS DATA
    # -------------------------------------------------------------------------

    return (
        masked_longitudinal_data,
        group_assignments,
        group_mask
    )