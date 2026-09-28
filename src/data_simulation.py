"""
Longitudinal Data Simulation
============================

This module defines the data-generating mechanism used in the longitudinal
graph neural network simulation experiments.

The simulation produces two related data sets:

1. longitudinal_data
   One row per patient visit containing irregular observation times and
   five longitudinal measurements.

2. patient_data
   One row per patient containing patient-level latent variables and
   prediction outcomes.

The longitudinal measurements generated here are COMPLETE. Missing values
are introduced separately by the missing-data generation module so that
the true unobserved values remain available for evaluating imputation
methods.

Both a binary classification outcome and a continuous regression outcome
are generated from the same underlying patient-level latent variables.
"""


import numpy as np
import pandas as pd


# =============================================================================
# SIMULATE COMPLETE LONGITUDINAL DATA
# =============================================================================

def simulate_longitudinal_data(
    n_patients=1000,
    min_visits=6,
    max_visits=10,
    regression_noise_sd=1.0,
    seed=100
):
    """
    Simulate complete irregular longitudinal data for a cohort of patients.

    Each patient has a patient-specific number of visits and irregular
    observation times. Two latent patient-level variables influence the
    longitudinal disease process:

        z_i ~ Poisson(1)

    controls the number of sick visits, while

        v_i ~ Normal(0, 1)

    controls the patient's disease severity.

    Five longitudinal measurements are generated. The measurements follow
    mean-reverting trajectories, with measurements 2 through 5 responding
    differently during sick visits. Measurement 1 intentionally has no
    direct sickness effect.

    Two patient-level outcomes are generated:

    classification_outcome
        Binary indicator equal to 1 when z_i > 0.

    regression_outcome
        Continuous outcome generated from z_i, v_i, and independent
        Gaussian noise.

    Parameters
    ----------
    n_patients : int
        Number of patients to simulate.

    min_visits : int
        Minimum number of visits for each patient.

    max_visits : int
        Maximum number of visits for each patient.

    regression_noise_sd : float
        Standard deviation of the random noise added to the continuous
        regression outcome.

    seed : int
        Random seed used to make the simulation reproducible.

    Returns
    -------
    longitudinal_data : pandas.DataFrame
        Complete longitudinal data with one row per patient visit.

    patient_data : pandas.DataFrame
        Patient-level data containing latent variables and both prediction
        outcomes.
    """

    # -------------------------------------------------------------------------
    # 1. INITIALIZE RANDOM NUMBER GENERATOR
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(seed)


    # -------------------------------------------------------------------------
    # 2. GENERATE NUMBER OF VISITS FOR EACH PATIENT
    # -------------------------------------------------------------------------

    # Each patient receives an integer number of visits between
    # min_visits and max_visits, inclusive.

    number_of_visits = rng.integers(
        low=min_visits,
        high=max_visits + 1,
        size=n_patients
    )


    # -------------------------------------------------------------------------
    # 3. GENERATE PATIENT-LEVEL LATENT VARIABLES
    # -------------------------------------------------------------------------

    # z_i controls the potential number of sick visits for patient i.
    #
    #     z_i ~ Poisson(1)

    latent_sick_count = rng.poisson(
        lam=1.0,
        size=n_patients
    )


    # v_i controls patient-specific sickness intensity.
    #
    #     v_i ~ Normal(0, 1)

    latent_sickness_intensity = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_patients
    )


    # Convert v_i to a severity score between 0 and 1 using the
    # logistic function.

    sickness_severity = (
        1.0
        /
        (
            1.0
            + np.exp(-latent_sickness_intensity)
        )
    )


    # A patient cannot have more sick visits than total visits.

    number_of_sick_visits = np.minimum(
        latent_sick_count,
        number_of_visits
    )


    # -------------------------------------------------------------------------
    # 4. GENERATE PATIENT-LEVEL OUTCOMES
    # -------------------------------------------------------------------------

    # Binary classification outcome:
    #
    #     Y_i = 1 if z_i > 0
    #           0 otherwise

    classification_outcome = (
        latent_sick_count > 0
    ).astype(int)


    # Continuous regression outcome:
    #
    #     Y_i = 2 z_i + 1.5 v_i + epsilon_i
    #
    # where
    #
    #     epsilon_i ~ Normal(0, regression_noise_sd^2)
    #
    # Because z_i and v_i also affect the longitudinal disease process,
    # prediction of this outcome requires the model to recover information
    # about the patient's latent state from the longitudinal measurements.

    regression_noise = rng.normal(
        loc=0.0,
        scale=regression_noise_sd,
        size=n_patients
    )

    regression_outcome = (
        2.0 * latent_sick_count
        + 1.5 * latent_sickness_intensity
        + regression_noise
    )


    # -------------------------------------------------------------------------
    # 5. CREATE PATIENT-LEVEL DATA FRAME
    # -------------------------------------------------------------------------

    patient_data = pd.DataFrame({
        "patient_id": np.arange(
            1,
            n_patients + 1
        ),
        "number_of_visits": number_of_visits,
        "latent_sick_count": latent_sick_count,
        "latent_sickness_intensity": latent_sickness_intensity,
        "sickness_severity": sickness_severity,
        "number_of_sick_visits": number_of_sick_visits,
        "classification_outcome": classification_outcome,
        "regression_outcome": regression_outcome
    })


    # -------------------------------------------------------------------------
    # 6. CREATE ONE ROW FOR EACH PATIENT VISIT
    # -------------------------------------------------------------------------

    visit_rows = []

    for patient_index in range(n_patients):

        patient_id = patient_index + 1

        for visit_number in range(
            1,
            number_of_visits[patient_index] + 1
        ):

            visit_rows.append({
                "patient_id": patient_id,
                "visit_number": visit_number
            })


    longitudinal_data = pd.DataFrame(
        visit_rows
    )


    # -------------------------------------------------------------------------
    # 7. ASSIGN SICK VISITS
    # -------------------------------------------------------------------------

    # Every visit initially begins as a healthy visit.

    longitudinal_data["sick_visit_indicator"] = 0


    # For each patient, randomly select the required number of sick visits.

    for patient_index in range(n_patients):

        patient_id = patient_index + 1

        patient_rows = longitudinal_data.index[
            longitudinal_data["patient_id"] == patient_id
        ].to_numpy()

        n_sick = number_of_sick_visits[
            patient_index
        ]


        if n_sick > 0:

            sick_rows = rng.choice(
                patient_rows,
                size=n_sick,
                replace=False
            )

            longitudinal_data.loc[
                sick_rows,
                "sick_visit_indicator"
            ] = 1


    # -------------------------------------------------------------------------
    # 8. GENERATE IRREGULAR TIME GAPS
    # -------------------------------------------------------------------------

    # Every patient begins at time zero.

    longitudinal_data[
        "time_since_previous_visit"
    ] = 0.0


    # Later visit gaps follow exponential distributions.
    #
    # Sick visits tend to occur after shorter intervals, with this effect
    # increasing as patient-specific sickness severity increases.

    for patient_index in range(n_patients):

        patient_id = patient_index + 1

        patient_rows = longitudinal_data.index[
            longitudinal_data["patient_id"] == patient_id
        ].to_numpy()


        # The first visit remains at time zero, so we start at the
        # second visit.

        for visit_index in range(
            1,
            len(patient_rows)
        ):

            row = patient_rows[
                visit_index
            ]

            sick_visit = longitudinal_data.loc[
                row,
                "sick_visit_indicator"
            ]


            mean_time_gap = (
                2.0
                - 1.5
                * sickness_severity[patient_index]
                * sick_visit
            )


            longitudinal_data.loc[
                row,
                "time_since_previous_visit"
            ] = rng.exponential(
                scale=mean_time_gap
            )


    # Convert the visit-specific time gaps into cumulative observation times.

    longitudinal_data["time"] = (
        longitudinal_data
        .groupby("patient_id")[
            "time_since_previous_visit"
        ]
        .cumsum()
    )


    # -------------------------------------------------------------------------
    # 9. DEFINE LONGITUDINAL MEASUREMENT PARAMETERS
    # -------------------------------------------------------------------------

    number_of_measurements = 5

    measurement_names = [
        f"x{measurement_index}"
        for measurement_index in range(
            1,
            number_of_measurements + 1
        )
    ]


    # All measurements have long-term baseline mean 5.

    baseline_mean = np.full(
        number_of_measurements,
        5.0
    )


    # Mean-reversion parameter.

    mean_reversion = 0.5


    # Standard deviation of random longitudinal movement.

    longitudinal_noise_sd = 0.5


    # Effects of a sick visit on each of the five measurements.
    #
    # x1 intentionally has no direct sickness effect.

    sickness_effects = np.array([
        0.0,
        3.0,
        -2.0,
        1.5,
        0.75
    ])


    # Create empty measurement columns.

    for measurement_name in measurement_names:

        longitudinal_data[
            measurement_name
        ] = np.nan


    # -------------------------------------------------------------------------
    # 10. GENERATE FIRST-VISIT MEASUREMENTS
    # -------------------------------------------------------------------------

    # Each patient begins near the common baseline measurement vector.

    for patient_index in range(n_patients):

        patient_id = patient_index + 1

        first_row = longitudinal_data.index[
            longitudinal_data["patient_id"] == patient_id
        ][0]


        initial_measurements = (
            baseline_mean
            + rng.normal(
                loc=0.0,
                scale=0.5,
                size=number_of_measurements
            )
        )


        longitudinal_data.loc[
            first_row,
            measurement_names
        ] = initial_measurements


    # -------------------------------------------------------------------------
    # 11. GENERATE LONGITUDINAL MEASUREMENT TRAJECTORIES
    # -------------------------------------------------------------------------

    # Each visit depends on the previous visit.
    #
    # Measurements tend to move back toward the baseline mean while sick
    # visits create measurement-specific departures from that trajectory.

    for patient_index in range(n_patients):

        patient_id = patient_index + 1

        patient_rows = longitudinal_data.index[
            longitudinal_data["patient_id"] == patient_id
        ].to_numpy()


        for visit_index in range(
            1,
            len(patient_rows)
        ):

            previous_row = patient_rows[
                visit_index - 1
            ]

            current_row = patient_rows[
                visit_index
            ]


            previous_measurements = longitudinal_data.loc[
                previous_row,
                measurement_names
            ].to_numpy(
                dtype=float
            )


            # Mean-reverting random change.

            random_change = rng.normal(
                loc=(
                    mean_reversion
                    * (
                        baseline_mean
                        - previous_measurements
                    )
                ),
                scale=longitudinal_noise_sd,
                size=number_of_measurements
            )


            current_measurements = (
                previous_measurements
                + random_change
            )


            # Add sickness effects when the current visit is sick.

            sick_visit = longitudinal_data.loc[
                current_row,
                "sick_visit_indicator"
            ]


            current_measurements += (
                sickness_effects
                * sickness_severity[
                    patient_index
                ]
                * sick_visit
            )


            longitudinal_data.loc[
                current_row,
                measurement_names
            ] = current_measurements


    # -------------------------------------------------------------------------
    # 12. RETURN COMPLETE SIMULATED DATA
    # -------------------------------------------------------------------------

    return (
        longitudinal_data,
        patient_data
    )