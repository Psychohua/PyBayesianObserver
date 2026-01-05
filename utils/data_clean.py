# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 15:48:00 2026

@author: LocalZ002
"""

import numpy as np


def data_clean(sample, response, option=2):
    """
    Exclude outliers from behavioral data

    Parameters
    ----------
    sample : array-like
        stimulus (seconds or milliseconds)
    response : array-like
        response (seconds or milliseconds)
    option : int
        1 = 3 SD rule
        2 = MAD within each stimulus (Remington et al.)
        3 = interquartile rule

    Returns
    -------
    sample_clean : ndarray
    response_clean : ndarray
    sti : ndarray
        unique stimulus values
    exclude_ind : ndarray
        indices of excluded trials
    """

    sample = np.asarray(sample).astype(float)
    response = np.asarray(response).astype(float)

    # ---- convert ms -> s ----
    if np.mean(sample) > 100:
        sample = sample / 1000.0
    if np.mean(response) > 100:
        response = response / 1000.0

    TS = sample.copy()
    TP = response.copy()

    exclude_ind = []

    # =========================
    # option 1: 3 SD
    # =========================
    if option == 1:
        bias = np.abs(TP - TS)
        exclude_ind = np.where(bias > 3 * np.std(bias))[0]

    # =========================
    # option 2: MAD by stimulus
    # =========================
    elif option == 2:
        sti = np.unique(TS)
        for s in sti:
            ind = np.where(TS == s)[0]
            res_temp = TP[ind]
            temp_bias = np.abs(res_temp - np.mean(res_temp))
            mad = np.median(temp_bias)
            out_ind = ind[temp_bias > 3 * mad]
            exclude_ind.extend(out_ind.tolist())

        exclude_ind = np.array(exclude_ind, dtype=int)

    # =========================
    # option 3: IQR
    # =========================
    elif option == 3:
        q75, q25 = np.percentile(TP, [75, 25])
        iqr_num = q75 - q25
        exclude_ind = np.where(TP > 3 * iqr_num)[0]

    else:
        raise ValueError("option must be 1, 2, or 3")

    # ---- remove outliers ----
    sample_clean = np.delete(TS, exclude_ind)
    response_clean = np.delete(TP, exclude_ind)

    sti = np.unique(sample_clean)

    return sample_clean, response_clean, sti, exclude_ind