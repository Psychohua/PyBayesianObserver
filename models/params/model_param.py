# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:07:47 2025

# models/params/model_param.py

@author: Huawei He
"""

import numpy as np
# from scipy.stats import uniform, norm

class ModelParam:
    """
    Python equivalent of MATLAB ModelParam
    """

    def __init__(self, value=np.nan, prior=None, isNonNegative=False):
        """
        Parameters
        ----------
        value : float
            Initial value of parameter
        prior : scipy.stats distribution
            Prior distribution
        isNonNegative : bool
            Whether parameter is constrained to be non-negative
        """
        self.prior = prior
        self.isNonNegative = isNonNegative

        # private storage (equivalent to realValue)
        self._realValue = value

    # =====================================================
    # Dependent properties
    # =====================================================

    @property
    def priorSupport(self):
        """
        Reasonable range for parameter
        """
        if self.prior is None:
            return None

        # Uniform prior
        if self.prior.dist.name == "uniform":
            lower = self.prior.kwds.get("loc", 0)
            upper = lower + self.prior.kwds.get("scale", 1)
            return np.array([lower, upper])

        # Normal prior
        if self.prior.dist.name == "norm":
            mu = self.prior.kwds.get("loc", 0)
            sigma = self.prior.kwds.get("scale", 1)
            return np.array([mu - 2.5 * sigma, mu + 2.5 * sigma])

        raise NotImplementedError("Unsupported prior distribution")

    @property
    def value(self):
        """
        Actual value (with non-negativity constraint if required)
        """
        if self.isNonNegative:
            return abs(self._realValue)
        else:
            return self._realValue

    @value.setter
    def value(self, valueIn):
        """
        Set raw parameter value
        """
        self._realValue = valueIn
