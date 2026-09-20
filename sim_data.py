import numpy as np
import pandas as pd


# ============================================================
# GENERATE SIMULATED LONGITUDINAL DATA
# ============================================================

def generate_sim_data(
    N=500,
    min_visits=3,
    max_visits=10,
    p=5,
    seed=100
):
    np.random.seed(seed)
    # GENERATE NUMBER OF VISITS PER PATIENT

    n_i = np.random.randint(
        min_visits,
        max_visits + 1,
        size=N
    )

    # 3. GENERATE PATIENT-LEVEL LATENT VARIABLES
    # z_i represents the number of sick visits a patient
    # would potentially have.
    # z_i ~ Poisson(1)

    z = np.random.poisson(
        lam=1,
        size=N
    )

    # v_i represents a patient-specific sickness intensity.
    # v_i ~ N(0, 1)

    v = np.random.normal(
        loc=0,
        scale=1,
        size=N
    )

    # 4. CREATE PATIENT-LEVEL DATA FRAME

    patient_data = pd.DataFrame({
        "subject": np.arange(1, N + 1),
        "n_visits": n_i,
        "z": z,
        "v": v
    })

    # 5. DETERMINE NUMBER OF SICK VISITS
    # A patient cannot have more sick visits than total visits.
    #     s_i = min(z_i, n_i)

    s_i = np.minimum(
        z,
        n_i
    )

    patient_data["n_sick"] = s_i


    # 6. CREATE ONE ROW PER PATIENT VISIT
    # We now convert the patient-level information into longitudinal form.
    # Each row represents one visit for one patient.

    rows = []
    for i in range(N):
        for j in range(n_i[i]):
            rows.append({
                "subject": i + 1,
                "visit": j + 1
            })

    data = pd.DataFrame(rows)

    # 7. ASSIGN SICK VISITS
    # Start by assuming every visit is healthy.
    # For each patient, randomly choose which visits are sick.
    data["sick_visit"] = 0
 
    for i in range(N):

        subject = i + 1

        # Find rows belonging to this patient.
        patient_rows = data.index[
            data["subject"] == subject
        ].to_numpy()

        # Number of sick visits for this patient.
        n_sick = s_i[i]

        if n_sick > 0:

            sick_rows = np.random.choice(
                patient_rows,
                size=n_sick,
                replace=False
            )

            data.loc[
                sick_rows,
                "sick_visit"
            ] = 1


    # 8. CONVERT LATENT INTENSITY INTO SEVERITY
    # We transform v_i using the logistic function:
    #                 exp(v_i)
    # severity_i = ----------------
    #               1 + exp(v_i)
    # This forces severity to lie between 0 and 1.

    severity = (
        np.exp(v)
        /
        (1 + np.exp(v))
    )

    patient_data["severity"] = severity

    # 9. GENERATE IRREGULAR TIME GAPS
    # Each patient begins at time 0.
    # Later visits have exponentially distributed gaps.
    # Sick visits tend to occur after shorter gaps.

    data["gap"] = 0.0

    for i in range(N):

        subject = i + 1

        patient_rows = data.index[
            data["subject"] == subject
        ].to_numpy()

        # Start at j = 1 because the first visit has gap = 0.
        for j in range(1, len(patient_rows)):

            row = patient_rows[j]

            sick = data.loc[
                row,
                "sick_visit"
            ]

            # Mean gap gets smaller when the visit is sick,
            # especially for patients with high severity.
            mean_gap = (
                2
                -
                1.5 * severity[i] * sick
            )

            data.loc[
                row,
                "gap"
            ] = np.random.exponential(
                scale=mean_gap
            )


    # 10. CONVERT GAPS INTO ACTUAL VISIT TIMES
    # Example:
    # gaps: 0, 2, 1, 3
    # times: 0, 2, 3, 6

    data["time"] = (
        data.groupby("subject")["gap"]
        .cumsum()
    )

    # 11. DEFINE BASELINE FEATURE VALUES
    # We currently use five longitudinal features: x1, x2, x3, x4, x5
    # Their long-term baseline mean is 5.

    mu = np.full(
        p,
        5.0
    )

    # alpha controls mean reversion.
    # sigma_e controls random noise in the longitudinal trajectories.

    alpha = 0.5
    sigma_e = 0.5


    # Create empty feature columns.

    for k in range(1, p + 1):

        data[f"x{k}"] = np.nan

    # 12. GENERATE FIRST VISIT FEATURES
    # Every patient begins near the baseline vector mu.

    for i in range(N):

        subject = i + 1

        first_row = data.index[
            data["subject"] == subject
        ][0]

        data.loc[
            first_row,
            [f"x{k}" for k in range(1, p + 1)]
        ] = (
            mu
            +
            np.random.normal(
                loc=0,
                scale=0.5,
                size=p
            )
        )

    # 13. GENERATE LONGITUDINAL FEATURE TRAJECTORIES
    # Each visit depends on the previous visit.
    # Features tend to move back toward the baseline mu,
    # while sick visits create additional changes.

    feature_names = [
        f"x{k}"
        for k in range(1, p + 1)
    ]

    for i in range(N):

        subject = i + 1

        patient_rows = data.index[
            data["subject"] == subject
        ].to_numpy()

        for j in range(1, len(patient_rows)):

            previous_row = patient_rows[j - 1]
            current_row = patient_rows[j]

            # Previous feature values.
            previous_x = data.loc[
                previous_row,
                feature_names
            ].to_numpy(dtype=float)

            # Random mean-reverting change.
            epsilon = np.random.normal(
                loc=alpha * (mu - previous_x),
                scale=sigma_e,
                size=p
            )

            # Determine whether the current visit is sick.
            sick = data.loc[
                current_row,
                "sick_visit"
            ]

            # Begin with previous values plus random movement.
            current_x = previous_x + epsilon

            # SICKNESS EFFECTS
            # x1 is intentionally a "dud" feature.
            # It has no direct sickness effect.
            # x2 through x5 respond differently to sickness.

            current_x[1] += (
                3.0
                * severity[i]
                * sick
            )

            current_x[2] -= (
                2.0
                * severity[i]
                * sick
            )

            current_x[3] += (
                1.5
                * severity[i]
                * sick
            )

            current_x[4] += (
                0.75
                * severity[i]
                * sick
            )

            # Save feature values for this visit.
            data.loc[
                current_row,
                feature_names
            ] = current_x

    # 14. CREATE PATIENT-LEVEL BINARY OUTCOME
    # y_i = 1 if the patient had at least one sick visit.
    # y_i = 0 otherwise.

    y = (
        z > 0
    ).astype(int)

    patient_data["y"] = y

    # 15. RETURN BOTH DATA SETS
    # data: one row per patient visit
    # patient_data: one row per patient

    return data, patient_data