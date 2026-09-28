"""
Patient-Level Data Splitting
============================

This module defines the patient-level train, validation, and test splits used
throughout the longitudinal graph neural network experiments.

The patient split is generated ONCE and should then be reused for every
version of the simulated data, including:

    - complete longitudinal data
    - MCAR longitudinal data
    - group-specific masked data
    - group-specific masked data with additional MCAR missingness
    - imputed data
    - baseline GCN analyses
    - LongGCN analyses
    - classification tasks
    - regression tasks

Creating the split at the patient level ensures that all longitudinal
observations belonging to a patient remain in the same data partition.

It also ensures that every model comparison uses exactly the same patients
for training, validation, and testing.
"""


import numpy as np
import pandas as pd


# =============================================================================
# CREATE PATIENT-LEVEL SPLIT ASSIGNMENTS
# =============================================================================

def create_patient_split_assignments(
    patient_data,
    train_proportion=0.70,
    validation_proportion=0.15,
    test_proportion=0.15,
    seed=100
):
    """
    Randomly assign patients to training, validation, and test sets.

    The split is performed using patient identifiers rather than individual
    longitudinal observations. This ensures that every visit belonging to a
    patient remains in the same data partition.

    The resulting assignment table can be saved and reused across every
    version of the longitudinal data.

    Parameters
    ----------
    patient_data : pandas.DataFrame
        Patient-level data containing one row per patient and a patient_id
        column.

    train_proportion : float
        Proportion of patients assigned to the training set.

    validation_proportion : float
        Proportion of patients assigned to the validation set.

    test_proportion : float
        Proportion of patients assigned to the test set.

    seed : int
        Random seed used to generate the patient split.

    Returns
    -------
    split_assignments : pandas.DataFrame
        Data frame containing patient_id and the corresponding data split.
    """

    # -------------------------------------------------------------------------
    # 1. INITIALIZE RANDOM NUMBER GENERATOR
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )


    # -------------------------------------------------------------------------
    # 2. EXTRACT PATIENT IDENTIFIERS
    # -------------------------------------------------------------------------

    patient_ids = (
        patient_data[
            "patient_id"
        ]
        .to_numpy()
        .copy()
    )


    number_of_patients = len(
        patient_ids
    )


    # -------------------------------------------------------------------------
    # 3. RANDOMLY SHUFFLE PATIENTS
    # -------------------------------------------------------------------------

    rng.shuffle(
        patient_ids
    )


    # -------------------------------------------------------------------------
    # 4. DETERMINE SPLIT SIZES
    # -------------------------------------------------------------------------

    number_training = int(
        train_proportion
        * number_of_patients
    )


    number_validation = int(
        validation_proportion
        * number_of_patients
    )


    # Any patients remaining after the training and validation sets are
    # assigned to the test set.

    number_test = (
        number_of_patients
        - number_training
        - number_validation
    )


    # -------------------------------------------------------------------------
    # 5. ASSIGN PATIENTS TO EACH DATA PARTITION
    # -------------------------------------------------------------------------

    training_ids = patient_ids[
        :number_training
    ]


    validation_ids = patient_ids[
        number_training:
        number_training + number_validation
    ]


    test_ids = patient_ids[
        number_training
        + number_validation:
    ]


    # -------------------------------------------------------------------------
    # 6. CREATE PATIENT SPLIT ASSIGNMENT TABLE
    # -------------------------------------------------------------------------

    training_assignments = pd.DataFrame({
        "patient_id": training_ids,
        "data_split": "train"
    })


    validation_assignments = pd.DataFrame({
        "patient_id": validation_ids,
        "data_split": "validation"
    })


    test_assignments = pd.DataFrame({
        "patient_id": test_ids,
        "data_split": "test"
    })


    split_assignments = pd.concat(
        [
            training_assignments,
            validation_assignments,
            test_assignments
        ],
        ignore_index=True
    )


    # Sort by patient identifier so that the saved assignment table is
    # easier to inspect.

    split_assignments = (
        split_assignments
        .sort_values(
            "patient_id"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 7. RETURN PATIENT SPLIT ASSIGNMENTS
    # -------------------------------------------------------------------------

    return split_assignments


# =============================================================================
# APPLY PATIENT SPLITS TO LONGITUDINAL DATA
# =============================================================================

def split_longitudinal_data(
    longitudinal_data,
    split_assignments
):
    """
    Split longitudinal observations using previously created patient-level
    split assignments.

    Parameters
    ----------
    longitudinal_data : pandas.DataFrame
        Longitudinal data containing a patient_id column.

    split_assignments : pandas.DataFrame
        Patient-level assignment table containing patient_id and data_split.

    Returns
    -------
    training_data : pandas.DataFrame
        Longitudinal observations belonging to training patients.

    validation_data : pandas.DataFrame
        Longitudinal observations belonging to validation patients.

    test_data : pandas.DataFrame
        Longitudinal observations belonging to test patients.
    """

    # -------------------------------------------------------------------------
    # 1. ATTACH PATIENT SPLIT LABELS
    # -------------------------------------------------------------------------

    data_with_splits = longitudinal_data.merge(
        split_assignments,
        on="patient_id",
        how="left"
    )


    # -------------------------------------------------------------------------
    # 2. CREATE TRAINING DATA
    # -------------------------------------------------------------------------

    training_data = (
        data_with_splits.loc[
            data_with_splits[
                "data_split"
            ] == "train"
        ]
        .drop(
            columns="data_split"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 3. CREATE VALIDATION DATA
    # -------------------------------------------------------------------------

    validation_data = (
        data_with_splits.loc[
            data_with_splits[
                "data_split"
            ] == "validation"
        ]
        .drop(
            columns="data_split"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 4. CREATE TEST DATA
    # -------------------------------------------------------------------------

    test_data = (
        data_with_splits.loc[
            data_with_splits[
                "data_split"
            ] == "test"
        ]
        .drop(
            columns="data_split"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 5. RETURN LONGITUDINAL DATA PARTITIONS
    # -------------------------------------------------------------------------

    return (
        training_data,
        validation_data,
        test_data
    )


# =============================================================================
# APPLY PATIENT SPLITS TO PATIENT-LEVEL DATA
# =============================================================================

def split_patient_data(
    patient_data,
    split_assignments
):
    """
    Split patient-level data using previously created patient assignments.

    This is useful when patient-level outcomes or latent simulation variables
    are needed separately for the training, validation, and test sets.

    Parameters
    ----------
    patient_data : pandas.DataFrame
        Patient-level data containing one row per patient.

    split_assignments : pandas.DataFrame
        Patient-level assignment table containing patient_id and data_split.

    Returns
    -------
    training_patient_data : pandas.DataFrame
        Patient-level data for training patients.

    validation_patient_data : pandas.DataFrame
        Patient-level data for validation patients.

    test_patient_data : pandas.DataFrame
        Patient-level data for test patients.
    """

    # -------------------------------------------------------------------------
    # 1. ATTACH PATIENT SPLIT LABELS
    # -------------------------------------------------------------------------

    data_with_splits = patient_data.merge(
        split_assignments,
        on="patient_id",
        how="left"
    )


    # -------------------------------------------------------------------------
    # 2. CREATE TRAINING PATIENT DATA
    # -------------------------------------------------------------------------

    training_patient_data = (
        data_with_splits.loc[
            data_with_splits[
                "data_split"
            ] == "train"
        ]
        .drop(
            columns="data_split"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 3. CREATE VALIDATION PATIENT DATA
    # -------------------------------------------------------------------------

    validation_patient_data = (
        data_with_splits.loc[
            data_with_splits[
                "data_split"
            ] == "validation"
        ]
        .drop(
            columns="data_split"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 4. CREATE TEST PATIENT DATA
    # -------------------------------------------------------------------------

    test_patient_data = (
        data_with_splits.loc[
            data_with_splits[
                "data_split"
            ] == "test"
        ]
        .drop(
            columns="data_split"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 5. RETURN PATIENT DATA PARTITIONS
    # -------------------------------------------------------------------------

    return (
        training_patient_data,
        validation_patient_data,
        test_patient_data
    )