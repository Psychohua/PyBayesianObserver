import numpy as np
import glob
from scipy.io import loadmat
from scipy.stats import norm
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.io as sio
import warnings
import sys
PKG_DIR = Path(__file__).resolve().parents[1]  # PyBayesianObserver/
sys.path.insert(0, str(PKG_DIR))
data_dir = PKG_DIR / "datafiles"
save_dir = PKG_DIR / "results"

from models.observers.scalar_mmse import ScalarMmseObserver
from utils.data_clean import data_clean
from utils.data_stat import data_stat
warnings.filterwarnings('ignore')

files = sorted(glob.glob(f"{data_dir}/RSG*.mat"))

n_subject = len(files)

# =========================
fit_goodness = []
est_params_iden = []
stat_iden = []
stat_sim_all = []
simulate_data = {}

# =========================
# define prior (MATLAB makedist)
# =========================
prior = norm(loc=(1 + 0.6) / 2, scale=0.01)

# =========================
# main loop
# =========================
for dd, fname in enumerate(files, start=1):

    print(f"Fitting participant {dd}")

    mat = loadmat(fname)

    # ---- load data ----
    sample = mat["sample"].squeeze()
    response = mat["response"].squeeze()

    # ------------------------------------------------
    sample, response, sti, exclude_ind = data_clean(sample, response)

    if len(exclude_ind) / len(sample) >= 0.2:
        print(f"Participant {dd} should be removed")

    # ------------------------------------------------
    # data_stat (bias / std / RMSE)
    # ------------------------------------------------
    bias_tol, std_tol, rmse = data_stat(sample, response, sti)
    print(f"Bias: {bias_tol:.4f}, sqrt Var: {std_tol:.4f}, RMSE: {rmse:.4f}")

    stat_iden.append([rmse, bias_tol, std_tol])

    # ------------------------------------------------
    # model definition
    # ------------------------------------------------
    observer = ScalarMmseObserver()
    observer.sti_range = np.array([0.6, 1.0])
    observer.prior = prior

    # ------------------------------------------------
    # fitting
    # ------------------------------------------------
    fitOut = observer.FitData(
        sample,
        response,
        minSearchTries=20
    )

    # parameters
    params = np.array([getattr(observer, p).value for p in observer.paramNames])
    est_params_iden.append(params)

    # fit goodness
    fit_goodness.append([
        fitOut["logLikelihood"],
        fitOut["bic"],
        fitOut["aic"]
    ])

    # ------------------------------------------------
    # simulation (100 trials per stimulus)
    # ------------------------------------------------
    unique_sample = np.unique(sample)
    sample_sim = np.repeat(unique_sample, 100)

    r_sim = observer.SimTrial(sample_sim)

    simulate_data[f"par{dd}"] = {
        "sample": sample_sim,
        "response": r_sim
    }

    bias_sim, std_sim, rmse_sim = data_stat(
        sample_sim, r_sim, unique_sample
    )

    stat_sim_all.append([rmse_sim, bias_sim, std_sim])


# =========================
# convert to numpy arrays
# =========================
fit_goodness = np.asarray(fit_goodness)
est_params_iden = np.asarray(est_params_iden)
stat_iden = np.asarray(stat_iden)
stat_sim_all = np.asarray(stat_sim_all)

# =========================
# save
# =========================
np.savez(
    f"{save_dir}/param_Remington_Gaussian.npz",
    est_params_iden=est_params_iden,
    stat_iden=stat_iden,
    fit_goodness=fit_goodness,
    simulate_data=simulate_data,
    stat_sim_all=stat_sim_all
)


# =========================
# load and plot
# =========================
param_file = save_dir / "param_Remington_Gaussian.npz"
data = np.load(param_file, allow_pickle=True)

simulate_data   = data["simulate_data"].item()
fit_goodness    = data["fit_goodness"]
est_params_iden = data["est_params_iden"]

# =========================
# list experimental files
# =========================
files = sorted(data_dir.glob("RSG*.mat"))


fig = plt.figure(figsize=(18, 8), facecolor="white")

for dd, fname in enumerate(files, start=1):

    # -----------------------------------------------------
    # load empirical data
    # -----------------------------------------------------
    mat = sio.loadmat(fname, squeeze_me=True)
    sample = mat["sample"]
    response = mat["response"]

    sample, response, _, _ = data_clean(sample, response, option=2)

    # -----------------------------------------------------
    # empirical: mean ± SEM
    # -----------------------------------------------------
    sampleLevels, idx = np.unique(sample, return_inverse=True)
    nCond = len(sampleLevels)

    respMean = np.zeros(nCond)
    respSEM = np.zeros(nCond)

    for c in range(nCond):
        curResp = response[idx == c]
        respMean[c] = np.mean(curResp)
        respSEM[c] = np.std(curResp, ddof=1) / np.sqrt(len(curResp))

    ax = fig.add_subplot(2, 6, dd)

    mainColor = np.array([0.2, 0.45, 0.75])
    markerSize = 5

    ax.errorbar(
        sampleLevels,
        respMean,
        yerr=respSEM,
        fmt="o",
        color=mainColor,
        markerfacecolor=mainColor,
        markeredgecolor="k",
        markersize=markerSize,
        capsize=10,
        linestyle="none"
    )

    # identity line
    lims = [
        min(sampleLevels.min(), respMean.min()),
        max(sampleLevels.max(), respMean.max())
    ]
    ax.plot(lims, lims, "--", color=[0.4, 0.4, 0.4], linewidth=1.5)

    # -----------------------------------------------------
    # simulated data: mean ± SD
    # -----------------------------------------------------
    sim_key = f"par{dd}"
    sim = simulate_data[sim_key]

    sample_sim = sim["sample"]
    response_sim = sim["response"]

    sampleLevels_sim, idx_sim = np.unique(sample_sim, return_inverse=True)

    respMean_sim = np.zeros(len(sampleLevels_sim))
    respSD_sim = np.zeros(len(sampleLevels_sim))

    for c in range(len(sampleLevels_sim)):
        curResp = response_sim[idx_sim == c]
        respMean_sim[c] = np.mean(curResp)
        respSD_sim[c] = np.std(curResp, ddof=1)

    lineColor = np.array([0.55, 0.70, 0.90])
    alphaLevel = 0.30

    xShade = np.concatenate([sampleLevels_sim, sampleLevels_sim[::-1]])
    yShade = np.concatenate([
        respMean_sim - respSD_sim,
        (respMean_sim + respSD_sim)[::-1]
    ])

    ax.fill(
        xShade,
        yShade,
        color=lineColor,
        alpha=alphaLevel,
        edgecolor="none"
    )

    ax.plot(
        sampleLevels_sim,
        respMean_sim,
        "-",
        color=lineColor,
        linewidth=2
    )


    ax.set_xlabel("Sample duration (s)", fontsize=12)
    ax.set_ylabel("Reported duration (s)", fontsize=12)
    ax.set_title(f"subject {dd}", fontsize=12)

    ax.tick_params(direction="out", width=1.2, labelsize=11)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()

plt.tight_layout()
plt.show()





