"""
MICE3D Imputation Utilities
===========================

Utilities for applying the installed MICE3D package to the simulated
longitudinal datasets.

MICE3D expects data with shape:

    patients x time positions x measurements

The simulation data are stored in visit-level form, with patients having
different numbers of visits. These utilities therefore:

    1. Convert visit-level data to a padded three-dimensional array.
    2. Run MICE3D exactly as implemented by the installed package.
    3. Discard imputed values corresponding to padded visits.
    4. Restore the original patient IDs, visit numbers, and irregular times.
    5. Optionally restore a target missingness structure.

The target missingness structure is particularly important for the comparison:

    group-specific + MCAR
        ->
    MICE3D
        ->
    group-specific

MICE3D is first allowed to impute the input dataset normally. Structural
missing values are then restored using the original group-specific dataset,
so only the additional MCAR missingness remains imputed.
"""


from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning

from imputers.mice_3d import MICE3D


# =============================================================================
# MICE3D ARRAY REPRESENTATION
# =============================================================================

@dataclass
class MICE3DArrayData:
    """
    Padded representation supplied to MICE3D.
    """

    X: np.ndarray

    patient_ids: list

    patient_row_indices: list

    visit_mask: np.ndarray


# =============================================================================
# VALIDATE LONGITUDINAL DATA
# =============================================================================

def validate_longitudinal_data(
    longitudinal_data,
    measurement_columns,
    patient_column="patient_id",
    visit_column="visit_number",
    time_column="time"
):
    """
    Validate the columns required for MICE3D conversion.
    """

    required_columns = [

        patient_column,

        visit_column,

        time_column,

        *measurement_columns
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in longitudinal_data.columns
    ]


    if missing_columns:

        raise ValueError(
            "The longitudinal data are missing required columns: "
            f"{missing_columns}"
        )


    if len(longitudinal_data) == 0:

        raise ValueError(
            "The longitudinal dataset contains no rows."
        )


# =============================================================================
# CONVERT LONGITUDINAL DATA TO 3D ARRAY
# =============================================================================

def longitudinal_data_to_mice3d_array(
    longitudinal_data,
    measurement_columns,
    patient_column="patient_id",
    visit_column="visit_number",
    time_column="time"
):
    """
    Convert visit-level longitudinal data to a padded MICE3D array.
    """

    # -------------------------------------------------------------------------
    # 1. VALIDATE DATA
    # -------------------------------------------------------------------------

    validate_longitudinal_data(

        longitudinal_data=longitudinal_data,

        measurement_columns=measurement_columns,

        patient_column=patient_column,

        visit_column=visit_column,

        time_column=time_column
    )


    # -------------------------------------------------------------------------
    # 2. ORGANIZE PATIENT DATA
    # -------------------------------------------------------------------------

    patient_data_list = []


    for patient_id, patient_data in longitudinal_data.groupby(
        patient_column,
        sort=False
    ):

        patient_data = (
            patient_data
            .sort_values(
                [
                    visit_column,
                    time_column
                ]
            )
        )


        patient_data_list.append(
            (
                patient_id,
                patient_data
            )
        )


    # -------------------------------------------------------------------------
    # 3. DETERMINE ARRAY DIMENSIONS
    # -------------------------------------------------------------------------

    number_of_patients = len(
        patient_data_list
    )


    number_of_measurements = len(
        measurement_columns
    )


    maximum_number_of_visits = max(

        len(patient_data)

        for _, patient_data in patient_data_list
    )


    # -------------------------------------------------------------------------
    # 4. INITIALIZE PADDED ARRAY
    # -------------------------------------------------------------------------

    X = np.full(

        (
            number_of_patients,
            maximum_number_of_visits,
            number_of_measurements
        ),

        np.nan,

        dtype=np.float64
    )


    visit_mask = np.zeros(

        (
            number_of_patients,
            maximum_number_of_visits
        ),

        dtype=bool
    )


    patient_ids = []

    patient_row_indices = []


    # -------------------------------------------------------------------------
    # 5. INSERT PATIENT DATA
    # -------------------------------------------------------------------------

    for patient_position, (
        patient_id,
        patient_data

    ) in enumerate(
        patient_data_list
    ):

        number_of_visits = len(
            patient_data
        )


        X[
            patient_position,
            :number_of_visits,
            :
        ] = (
            patient_data[
                measurement_columns
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        visit_mask[
            patient_position,
            :number_of_visits
        ] = True


        patient_ids.append(
            patient_id
        )


        patient_row_indices.append(
            patient_data.index.to_numpy()
        )


    # -------------------------------------------------------------------------
    # 6. RETURN REPRESENTATION
    # -------------------------------------------------------------------------

    return MICE3DArrayData(

        X=X,

        patient_ids=patient_ids,

        patient_row_indices=patient_row_indices,

        visit_mask=visit_mask
    )


# =============================================================================
# RESTORE 3D ARRAY TO LONGITUDINAL DATA
# =============================================================================

def restore_mice3d_array_to_longitudinal_data(
    longitudinal_data,
    X_imputed,
    array_data,
    measurement_columns
):
    """
    Place MICE3D-imputed values back into the original longitudinal DataFrame.

    Padding positions are discarded.
    """

    if X_imputed.shape != array_data.X.shape:

        raise ValueError(
            "The imputed MICE3D array does not match the input shape."
        )


    imputed_longitudinal_data = (
        longitudinal_data.copy()
    )


    for patient_position, row_indices in enumerate(
        array_data.patient_row_indices
    ):

        number_of_visits = len(
            row_indices
        )


        patient_values = X_imputed[

            patient_position,

            :number_of_visits,

            :
        ]


        imputed_longitudinal_data.loc[
            row_indices,
            measurement_columns
        ] = patient_values


    return imputed_longitudinal_data


# =============================================================================
# RUN MICE3D
# =============================================================================

def impute_with_mice3d(
    longitudinal_data,
    measurement_columns,
    mice_max_iter=10,
    gp_alpha=0.001,
    random_state=None,
    patient_column="patient_id",
    visit_column="visit_number",
    time_column="time"
):
    """
    Apply MICE3D to one longitudinal dataset.

    The supplied dataset is treated as one independent MICE3D dataset.
    """

    # -------------------------------------------------------------------------
    # 1. CREATE 3D REPRESENTATION
    # -------------------------------------------------------------------------

    array_data = longitudinal_data_to_mice3d_array(

        longitudinal_data=longitudinal_data,

        measurement_columns=measurement_columns,

        patient_column=patient_column,

        visit_column=visit_column,

        time_column=time_column
    )


    # -------------------------------------------------------------------------
    # 2. RECORD ORIGINAL VALUES
    # -------------------------------------------------------------------------

    original_values = (
        longitudinal_data[
            measurement_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    missing_before = int(

        np.isnan(
            original_values
        )
        .sum()
    )


    total_measurement_cells = (
        original_values.size
    )


    # -------------------------------------------------------------------------
    # 3. CREATE MICE3D IMPUTER
    # -------------------------------------------------------------------------

    imputer = MICE3D(

        mice_max_iter=mice_max_iter,

        gp_alpha=gp_alpha,

        random_state=random_state
    )


    # -------------------------------------------------------------------------
    # 4. RUN MICE3D
    # -------------------------------------------------------------------------

    # MICE3D fits many Gaussian-process models internally.
    #
    # scikit-learn may repeatedly report that GP kernel parameters have reached
    # optimization bounds. These ConvergenceWarning messages are suppressed
    # without changing the MICE3D algorithm or fitted values.

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=ConvergenceWarning
        )


        X_imputed = imputer.fit_transform(
            array_data.X
        )


    # -------------------------------------------------------------------------
    # 5. RESTORE LONGITUDINAL DATAFRAME
    # -------------------------------------------------------------------------

    imputed_longitudinal_data = (
        restore_mice3d_array_to_longitudinal_data(

            longitudinal_data=longitudinal_data,

            X_imputed=X_imputed,

            array_data=array_data,

            measurement_columns=measurement_columns
        )
    )


    # -------------------------------------------------------------------------
    # 6. VERIFY OBSERVED VALUES DID NOT CHANGE
    # -------------------------------------------------------------------------

    imputed_values = (
        imputed_longitudinal_data[
            measurement_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    observed_mask = (
        ~np.isnan(
            original_values
        )
    )


    if not np.allclose(

        original_values[
            observed_mask
        ],

        imputed_values[
            observed_mask
        ]

    ):

        raise RuntimeError(
            "MICE3D changed one or more originally observed values."
        )


    # -------------------------------------------------------------------------
    # 7. CHECK FULL MICE3D OUTPUT
    # -------------------------------------------------------------------------

    missing_after_mice3d = int(

        np.isnan(
            imputed_values
        )
        .sum()
    )


    if missing_after_mice3d > 0:

        raise RuntimeError(
            "MICE3D left "
            f"{missing_after_mice3d} missing values at real visits."
        )


    # -------------------------------------------------------------------------
    # 8. DIAGNOSTICS
    # -------------------------------------------------------------------------

    diagnostics = {

        "number_of_patients":
            len(
                array_data.patient_ids
            ),

        "maximum_number_of_visits":
            array_data.X.shape[
                1
            ],

        "number_of_measurements":
            len(
                measurement_columns
            ),

        "total_measurement_cells":
            total_measurement_cells,

        "missing_before":
            missing_before,

        "missing_rate_before":
            (
                missing_before
                /
                total_measurement_cells
            ),

        "missing_after_mice3d":
            missing_after_mice3d
    }


    return (
        imputed_longitudinal_data,
        diagnostics
    )


# =============================================================================
# ALIGN REFERENCE DATA TO ANOTHER DATASET
# =============================================================================

def align_reference_data(
    data,
    reference_data,
    measurement_columns,
    patient_column="patient_id",
    visit_column="visit_number",
    time_column="time"
):
    """
    Align measurement values from a reference dataset to another dataset.
    """

    key_columns = [

        patient_column,

        visit_column,

        time_column
    ]


    reference_lookup = (
        reference_data
        .set_index(
            key_columns
        )[
            measurement_columns
        ]
    )


    if not reference_lookup.index.is_unique:

        raise ValueError(
            "The reference dataset does not have unique patient/visit keys."
        )


    data_index = pd.MultiIndex.from_frame(
        data[
            key_columns
        ]
    )


    missing_keys = (
        data_index
        .difference(
            reference_lookup.index
        )
    )


    if len(
        missing_keys
    ) > 0:

        raise ValueError(
            "The reference dataset is missing one or more "
            "patient/visit combinations."
        )


    aligned_reference = (
        reference_lookup
        .loc[
            data_index
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    return aligned_reference


# =============================================================================
# RESTORE TARGET MISSINGNESS STRUCTURE
# =============================================================================

def restore_target_missingness_structure(
    imputed_longitudinal_data,
    target_longitudinal_data,
    measurement_columns,
    patient_column="patient_id",
    visit_column="visit_number",
    time_column="time"
):
    """
    Restore missing cells required by a target longitudinal structure.

    Examples
    --------

    MCAR -> complete
        The complete target has no structural missing values, so nothing is
        restored.

    Group-specific + MCAR -> group-specific
        Missing cells in the group-specific target are restored to NaN after
        MICE3D finishes. The additional MCAR cells remain imputed.
    """

    result = (
        imputed_longitudinal_data.copy()
    )


    target_values = align_reference_data(

        data=result,

        reference_data=target_longitudinal_data,

        measurement_columns=measurement_columns,

        patient_column=patient_column,

        visit_column=visit_column,

        time_column=time_column
    )


    result_values = (
        result[
            measurement_columns
        ]
        .to_numpy(
            dtype=np.float64,
            copy=True
        )
    )


    target_missing_mask = (
        np.isnan(
            target_values
        )
    )


    result_values[
        target_missing_mask
    ] = np.nan


    result.loc[
        :,
        measurement_columns
    ] = result_values


    return result


# =============================================================================
# EVALUATE IMPUTATION AGAINST TARGET DATA
# =============================================================================

def evaluate_imputation_against_target(
    starting_longitudinal_data,
    imputed_longitudinal_data,
    target_longitudinal_data,
    measurement_columns,
    patient_column="patient_id",
    visit_column="visit_number",
    time_column="time"
):
    """
    Evaluate the cells MICE3D was intended to recover.

    A cell is evaluated when:

        starting dataset = missing

    and:

        target dataset = observed

    Therefore:

        MCAR -> complete

    evaluates the MCAR cells, while:

        group-specific + MCAR -> group-specific

    evaluates only the additional MCAR cells and excludes structural
    group-specific missingness.
    """

    # -------------------------------------------------------------------------
    # 1. GET ALIGNED VALUES
    # -------------------------------------------------------------------------

    starting_values = (
        starting_longitudinal_data[
            measurement_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    imputed_values = (
        imputed_longitudinal_data[
            measurement_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    target_values = align_reference_data(

        data=starting_longitudinal_data,

        reference_data=target_longitudinal_data,

        measurement_columns=measurement_columns,

        patient_column=patient_column,

        visit_column=visit_column,

        time_column=time_column
    )


    # -------------------------------------------------------------------------
    # 2. IDENTIFY CELLS THAT SHOULD HAVE BEEN IMPUTED
    # -------------------------------------------------------------------------

    evaluation_mask = (

        np.isnan(
            starting_values
        )

        &

        ~np.isnan(
            target_values
        )
    )


    number_imputed = int(
        evaluation_mask.sum()
    )


    if number_imputed == 0:

        raise ValueError(
            "No cells were identified for imputation evaluation."
        )


    # -------------------------------------------------------------------------
    # 3. OVERALL METRICS
    # -------------------------------------------------------------------------

    true_values = (
        target_values[
            evaluation_mask
        ]
    )


    estimated_values = (
        imputed_values[
            evaluation_mask
        ]
    )


    errors = (

        estimated_values

        -

        true_values
    )


    overall_metrics = {

        "number_imputed":
            number_imputed,

        "mae":
            float(
                np.mean(
                    np.abs(
                        errors
                    )
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            )
    }


    # -------------------------------------------------------------------------
    # 4. FEATURE-SPECIFIC METRICS
    # -------------------------------------------------------------------------

    feature_results = []


    for feature_position, measurement in enumerate(
        measurement_columns
    ):

        feature_mask = (
            evaluation_mask[
                :,
                feature_position
            ]
        )


        number_feature_imputed = int(
            feature_mask.sum()
        )


        if number_feature_imputed == 0:

            feature_mae = np.nan

            feature_rmse = np.nan


        else:

            feature_errors = (

                imputed_values[
                    feature_mask,
                    feature_position
                ]

                -

                target_values[
                    feature_mask,
                    feature_position
                ]
            )


            feature_mae = float(

                np.mean(
                    np.abs(
                        feature_errors
                    )
                )
            )


            feature_rmse = float(

                np.sqrt(
                    np.mean(
                        feature_errors ** 2
                    )
                )
            )


        feature_results.append(
            {
                "measurement":
                    measurement,

                "number_imputed":
                    number_feature_imputed,

                "mae":
                    feature_mae,

                "rmse":
                    feature_rmse
            }
        )


    feature_metrics = pd.DataFrame(
        feature_results
    )


    return (
        overall_metrics,
        feature_metrics
    )