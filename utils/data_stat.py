# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 15:48:47 2026

@author: LocalZ002
"""
import numpy as np

''
def data_stat(sample, response, sti):
    """
    Compute bias, variability (std), and RMSE
    Exactly matches MATLAB implementation

    Parameters
    ----------
    sample : ndarray
    response : ndarray
    sti : ndarray
        unique stimulus values

    Returns
    -------
    bias : float
    std : float
    RMSE : float
    """

    sample = np.asarray(sample)
    response = np.asarray(response)
    sti = np.asarray(sti)

    # =========================
    # bias
    # =========================
    bias_temp = []
    for s in sti:
        ind = np.where(sample == s)[0]
        b = np.mean(response[ind] - sample[ind]) ** 2
        bias_temp.append(b)

    tol_bias = np.sqrt(np.mean(bias_temp))

    # =========================
    # std (variability)
    # =========================
    std_temp = []
    for s in sti:
        ind = np.where(sample == s)[0]
        std_n = np.var(response[ind])
        std_temp.append(std_n)

    tol_std = np.sqrt(np.mean(std_temp))

    # =========================
    # RMSE
    # =========================
    RMSE = np.sqrt(tol_bias**2 + tol_std**2)

    return tol_bias, tol_std, RMSE