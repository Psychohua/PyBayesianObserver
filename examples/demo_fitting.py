# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:25:55 2025

Bayesian Obeserver Model

@author: Huawei He
"""
from pathlib import Path
import sys
PKG_DIR = Path(__file__).resolve().parents[1]  # PyBayesianObserver/
sys.path.insert(0, str(PKG_DIR))
# package import
import numpy as np
from models.observers.scalar_mmse import ScalarMmseObserver
import warnings
warnings.filterwarnings('ignore')

def main():

    np.random.seed(0)

    # -----------------------------
    # Create model
    # -----------------------------
    model = ScalarMmseObserver()

    # Ground-truth parameters
    model.w_m.value = 0.12
    model.w_p.value = 0.08
    model.offset.value = 0.02
    model.Sigma_Gau.value = 0.05

    # -----------------------------
    # Simulate data
    # -----------------------------
    nTrials = 200
    s = np.random.uniform(0.6, 1.0, nTrials)
    r = model.SimTrial(s)

    # -----------------------------
    # Reset parameters (important!)
    # -----------------------------
    model.RandomizeParamVal()

    print("Initial parameters:")
    for k in model.paramNames:
        print(f"  {k}: {model.__dict__[k].value:.3f}")

    # -----------------------------
    # Fit model
    # -----------------------------
    fitOut = model.FitData(
        s,
        r,
        minSearchTries=5,
        display="off"
    )

    # -----------------------------
    # Show results
    # -----------------------------
    print("\nRecovered parameters:")
    for k, v in fitOut["wFit"].items():
        print(f"  {k}: {v:.3f}")

    print("\nFit statistics:")
    print(f"  logLik: {fitOut['logLikelihood']:.2f}")
    print(f"  AIC   : {fitOut['aic']:.2f}")
    print(f"  BIC   : {fitOut['bic']:.2f}")


if __name__ == "__main__":
    main()
