# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:07:03 2025

# models/base/psych_model.py

@author: Huawei He
"""

from abc import ABC
import numpy as np
from scipy.stats import uniform, norm
from scipy.optimize import minimize

class PsychModel(ABC):
    """
    PsychModel
    """

    def __init__(self):
        # ===== properties =====
        self.prior = uniform(loc=0.5, scale=0.5)   # Uniform(0.5, 1)
        self.Transform = lambda a: a               # identity transform
        self.plotColor = None

    # =====================================================
    # Dependent properties
    # =====================================================

    @property
    def priorSupport(self):
        """
        Equivalent to get.priorSupport
        """
        # scipy.stats does not have DistributionName,
        # so we infer from the distribution type
        if self.prior.dist.name == "uniform":
            lower = self.prior.kwds.get("loc", 0)
            upper = lower + self.prior.kwds.get("scale", 1)
            return np.array([lower, upper])

        if self.prior.dist.name == "norm":
            # MATLAB code: support = obj.sti_range
            # assume subclass defines self.sti_range
            return self.sti_range

        raise NotImplementedError("Unknown prior distribution")

    # @property
    # def paramNames(self):
    #     """
    #     Find all attributes that are instances of ModelParam
    #     """
    #     from models.params.model_param import ModelParam
    #     names = []
    #     for attr in dir(self):
    #         # skip private / built-in
    #         if attr.startswith("_"):
    #             continue
    #         elif attr in ("params", "paramNames"):
    #             continue
    #         else:
    #             try:
    #                 value = getattr(self, attr)
    #             except AttributeError:
    #                 continue
    #             if isinstance(value, ModelParam):
    #                 names.append(attr)
    #     return names
    @property
    def paramNames(self):
        """
        Find all parameter-like attributes (robust to import issues)
        """
        names = []
        for attr, value in self.__dict__.items():
            if hasattr(value, "value") and hasattr(value, "prior"):
                names.append(attr)
        return names

    @property
    def params(self):
        """
        Equivalent to get.params → returns a table-like dict
        """
        values = {}
        for name in self.paramNames:
            values[name] = getattr(self, name).value
        return values

    # =====================================================
    # Methods
    # =====================================================

    def RandomizeParamVal(self, params=None):
        """
        Randomly sample parameter values from their priors
        """
        if params is None or len(params) == 0:
            params = self.paramNames

        for name in params:
            param = getattr(self, name)
            param.value = param.prior.rvs()

    def SetParamVal(self, params, vals=None):
        """
        Set parameter values
        """
        # MATLAB: table input
        if isinstance(params, dict):
            for k, v in params.items():
                getattr(self, k).value = v
            return

        # single param name
        if isinstance(params, str):
            params = [params]

        for i, name in enumerate(params):
            getattr(self, name).value = vals[i]

    # =====================================================
    # Sealed method
    # =====================================================

    # def FitData(self, s, r, *args, **kwargs):
    #     """
    #     Placeholder for FitData
    #     (do NOT override in subclasses)
    #     """
    #     raise NotImplementedError("FitData is implemented elsewhere")

    def GetEstimate(self, m, intMethod='simpson', nSamples=500):
        """
        Bayesian estimate of stimulus given measurement m
        """

        # -------------------------------------------------
        # Prior handling (matches MATLAB switch)
        # -------------------------------------------------
        # NOTE: MATLAB version overwrites obj.prior for Normal case
        # hasattr: to check if an object has a specific attribute
        if hasattr(self.prior, "dist") and self.prior.dist.name == "uniform":
            support = self.priorSupport

        elif hasattr(self.prior, "dist") and self.prior.dist.name == "norm":
            prior = norm(
                loc=np.mean(self.sti_range),
                scale=self.Sigma_Gau.value
            )
            support = self.sti_range
            self.prior = prior  # side effect preserved

        else:
            raise NotImplementedError("Unsupported prior distribution")

        # -------------------------------------------------
        # Numerical integration (Simpson's rule)
        # -------------------------------------------------
        if intMethod.lower() != "simpson":
            raise NotImplementedError("Only Simpson integration is supported")

        h = (support[1] - support[0]) / (nSamples - 1)
        s = np.linspace(support[0], support[1], nSamples)

        w = np.ones(nSamples)
        w[1:-1:2] = 4
        w[2:-2:2] = 2
        w = w * h / 3

        # -------------------------------------------------
        # Grid expansion (trial × grid)
        # -------------------------------------------------
        m = np.atleast_1d(m)
        nTrial = m.shape[0]
        nSamp = s.size

        s = np.tile(s, (nTrial, 1))
        m = np.tile(m.reshape(-1, 1), (1, nSamp))

        # -------------------------------------------------
        # Likelihood and posterior
        # -------------------------------------------------
        # P(tm | ts)
        l = self.ProbMeasure_Sample(m, s)

        # posterior = likelihood * prior
        p = l * self.prior.pdf(s)

        # -------------------------------------------------
        # Bayesian estimator: E[s | m]
        # -------------------------------------------------
        alpha = 1.0 / (w @ p.T)
        estimate_mean = (alpha * (w @ (s * p).T)).T

        return estimate_mean


    def _minFunProb(self, r, s, x, paramsToFit):
        """
        Equivalent to MATLAB subfunction minFunProb
        """
        self.SetParamVal(paramsToFit, x)

        if r.size == 0:
            return np.array([np.nan])

        out = self.ProbResponse_Sample(r, s)
        out = out[np.isfinite(out)]
        return out


    def FitData(
        self,
        s,
        r,
        display="off",
        paramsToFit=None,
        minSearchTries=1,
    ):


        if paramsToFit is None:
            paramsToFit = self.paramNames

        # -----------------------------------------
        # Preprocess data
        # -----------------------------------------
        s = np.asarray(s).reshape(-1)
        r = np.asarray(r).reshape(-1)

        mask = ~np.isnan(r)
        s = s[mask]
        r = r[mask]

        # -----------------------------------------
        # Objective function: negative log-likelihood
        # -----------------------------------------
        def minFun(x):
            return -np.sum(np.log(self._minFunProb(r, s, x, paramsToFit)))

        # -----------------------------------------
        # Multi-start optimization
        # -----------------------------------------
        fitVals = []
        negLogLikelihood = []
        numMaxEvals = 0

        for _ in range(minSearchTries):

            # ---- random initial guess from priors
            initialGuess = np.array([
                self.__dict__[p].prior.rvs() for p in paramsToFit
            ])

            # ---- bounds
            bounds = []
            for p in paramsToFit:
                param = self.__dict__[p]
                if param.isNonNegative:
                    lb = 0.0
                else:
                    lb = -1.0
                ub = 1.0
                bounds.append((lb, ub))

            # last param: prior SD
            bounds[-1] = (-np.inf, np.inf)

            # ---- optimizer
            res = minimize(
                minFun,
                initialGuess,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(
                    maxiter=3000 * len(paramsToFit),
                    disp=(display != "off")
                )
            )

            fitVals.append(res.x)
            negLogLikelihood.append(res.fun)

            if not res.success:
                numMaxEvals += 1

        fitVals = np.column_stack(fitVals)
        negLogLikelihood = np.array(negLogLikelihood)

        # -----------------------------------------
        # Best fit
        # -----------------------------------------
        bestFitInd = np.argmin(negLogLikelihood)
        bestNegLL = negLogLikelihood[bestFitInd]

        self.SetParamVal(paramsToFit, fitVals[:, bestFitInd])

        # -----------------------------------------
        # Model selection metrics
        # -----------------------------------------
        nParams = len(paramsToFit)

        fitOut = dict()
        fitOut["logLikelihood"] = bestNegLL
        fitOut["aic"] = 2 * bestNegLL + 2 * nParams
        fitOut["bic"] = 2 * bestNegLL + nParams * np.log(len(r))
        fitOut["fitTries"] = minSearchTries
        fitOut["fitFails"] = numMaxEvals

        fitOut["wFit"] = {}
        for p in self.paramNames:
            fitOut["wFit"][p] = self.__dict__[p].value

        # ---- status message
        if numMaxEvals > 0:
            print(f"Exceeded max evals {numMaxEvals} times ({self.__class__.__name__}).")
        else:
            print(f"Successful fit ({self.__class__.__name__}).")

        return fitOut
