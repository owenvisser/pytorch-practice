"""
LongGCN Data Preparation
========================

This module converts the simulated visit-level longitudinal data into the
observation-level long format expected by the LongGCN package.

The simulated data are stored with one row per patient visit and one column
for each longitudinal measurement:

    patient_id
    visit_number
    time
    x1
    x2
    x3
    x4
    x5

LongGCN instead represents each observed scalar measurement as its own row:

    patient_id
    time
    measurement
    value

Measurements that are missing are excluded from the resulting observation-
level data. Therefore, the same conversion function can be used for complete
data, MCAR data, group-specific masked data, and group-specific data with
additional MCAR missingness.

Graph construction and the generalized longitudinal matrix representation are
handled by the LongGCN package after this conversion.
"""


import pandas as pd


# =============================================================================
# CONVERT VISIT-LEVEL DATA TO LONGGCN OBSERVATION FORMAT
# =============================================================================

def convert_to_longgcn_format(
    longitudinal_data,
    measurement_columns=None
):
    """
    Convert visit-level longitudinal data to observation-level LongGCN format.

    Each observed scalar longitudinal measurement becomes one row in the
    output data set. Missing measurement values are omitted.

    Parameters
    ----------
    longitudinal_data : pandas.DataFrame
        Visit-level longitudinal data containing patient_id, visit_number,
        time, and the longitudinal measurement columns.

    measurement_columns : list of str or None
        Longitudinal measurement columns to include. If None, the default
        measurements x1 through x5 are used.

    Returns
    -------
    longgcn_data : pandas.DataFrame
        Observation-level longitudinal data containing:

            patient_id
            visit_number
            time
            measurement
            value

        Only observed measurement values are included.
    """

    # -------------------------------------------------------------------------
    # 1. DEFINE LONGITUDINAL MEASUREMENTS
    # -------------------------------------------------------------------------

    if measurement_columns is None:

        measurement_columns = [
            "x1",
            "x2",
            "x3",
            "x4",
            "x5"
        ]


    # -------------------------------------------------------------------------
    # 2. RESHAPE VISIT-LEVEL DATA INTO MEASUREMENT-LEVEL LONG FORMAT
    # -------------------------------------------------------------------------

    longgcn_data = longitudinal_data.melt(

        id_vars=[
            "patient_id",
            "visit_number",
            "time"
        ],

        value_vars=measurement_columns,

        var_name="measurement",

        value_name="value"
    )


    # -------------------------------------------------------------------------
    # 3. REMOVE UNOBSERVED MEASUREMENTS
    # -------------------------------------------------------------------------

    # LongGCN represents only measurements that were actually observed.
    #
    # Therefore, missing values do not appear as rows in the generalized
    # observation-level representation.

    longgcn_data = (
        longgcn_data
        .dropna(
            subset=[
                "value"
            ]
        )
    )


    # -------------------------------------------------------------------------
    # 4. SORT OBSERVATIONS
    # -------------------------------------------------------------------------

    # Sorting is not what defines the LongGCN structure, but keeping the
    # observations ordered makes the resulting data easier to inspect and
    # ensures reproducible saved files.

    longgcn_data = (
        longgcn_data
        .sort_values(
            by=[
                "patient_id",
                "time",
                "measurement"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # 5. RETURN LONGGCN-READY DATA
    # -------------------------------------------------------------------------

    return longgcn_data