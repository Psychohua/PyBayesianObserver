# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:07:27 2025

# models/observers/scalar_mmse.py

@author: Huawei He
"""

import numpy as np
from scipy.stats import norm
from models.base.psych_model import PsychModel
from models.params.model_param import ModelParam
# from scipy.optimize import minimize

class ScalarMmseObserver(PsychModel):
    """
    Python ScalarMmseObserver
    """

    def __init__(self):
        super().__init__()

        # ===== parameters =====
        self.w_m = ModelParam(
            prior=norm(loc=0.1, scale=0.02),
            isNonNegative=True
        )

        self.w_p = ModelParam(
            prior=norm(loc=0.1, scale=0.01),
            isNonNegative=True
        )

        self.offset = ModelParam(
            prior=norm(loc=0.0, scale=0.1)
        )

        self.Sigma_Gau = ModelParam(
            prior=norm(loc=0.1, scale=0.01),
            isNonNegative=True
        )

        # ===== flags =====
        self.observerActor = False
        self.sti_range = np.array([0.6, 1.0])

        # ===== constructor behavior =====
        self.plotColor = np.array([1.0, 0.0, 0.0])

        # prior over stimulus
        self.prior = norm(
            loc=np.mean(self.sti_range),
            scale=0.01
        )

    # =====================================================
    # Bayesian estimation
    # =====================================================

    def GetAim(self, m):
        """
        Bayesian estimate of stimulus
        """
        wm = self.w_m.value

        if wm == 0:
            e = m
        elif np.isinf(wm):
            e = np.mean(self.priorSupport) * np.ones_like(m)
        else:
            e = self.GetEstimate(m)   # implemented elsewhere (same as MATLAB)

        # systematic bias
        e = e + self.offset.value

        if self.observerActor:
            a = self.Transform(e) / (1.0 + self.w_p.value ** 2)
        else:
            a = self.Transform(e)

        return a

    # =====================================================
    # Noisy measurement
    # =====================================================

    def GetMeasurement(self, s):
        """
        Noisy sensory measurement
        """
        return np.random.randn(*np.shape(s)) * s * self.w_m.value + s

    # =====================================================
    # Noisy production
    # =====================================================

    def GetProduction(self, a):
        """
        Noisy motor production
        """
        return np.random.randn(*np.shape(a)) * a * self.w_p.value + a

    # =====================================================
    # Simulate a single trial
    # =====================================================

    def SimTrial(self, s):
        m = self.GetMeasurement(s)
        a = self.GetAim(m)
        r = self.GetProduction(a)
        return r

    # =====================================================
    # Likelihood components
    # =====================================================

    def ProbMeasure_Sample(self, m, s):
        """
        P(tm | ts)
        """
        sigma = s * self.w_m.value
        return norm.pdf(m, loc=s, scale=sigma)

    def ProbResponse_Aim(self, r, a):
        """
        P(tp | te)
        """
        sigma = a * self.w_p.value
        return norm.pdf(r, loc=a, scale=sigma)

    def ProbResponse_Sample(self, r, s):
        """
        P(tp | ts) via numerical integration
        """
        nIntPoints = 500
        support = self.priorSupport

        intBounds = np.array([
            max(1e-5, support[0] - 1 * self.w_m.value * support[0]),
            support[1] + 1 * self.w_m.value * support[1]
        ])

        # Simpson integration grid
        h = np.diff(intBounds)[0] / (nIntPoints - 1)
        m = np.linspace(intBounds[0], intBounds[1], nIntPoints)

        weights = np.ones(nIntPoints)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
        weights = weights * h / 3

        # mesh grids
        a = self.GetAim(m)
        r_grid, a_grid = np.meshgrid(r, a)
        s_grid, m_grid = np.meshgrid(s, m)

        integrand = (
            self.ProbResponse_Aim(r_grid, a_grid) *
            self.ProbMeasure_Sample(m_grid, s_grid)
        )

        p = (weights @ integrand).T
        return p



