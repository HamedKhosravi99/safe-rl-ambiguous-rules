"""Real prognostic features from NASA C-MAPSS (Domains FD001-FD004, grounded).

Replaces the synthetic feature-generating process in saorl.env with features
derived from the real turbofan run-to-failure benchmark. Each engine cycle gets:

  * rul_true : capped piecewise-linear remaining useful life (a construction/eval
               label only -- never a policy input), RUL_true = min(cycles_left, CAP);
  * rul_hat  : out-of-fold RandomForest RUL prediction from the informative
               sensors (a real, honestly cross-validated prognostic estimate);
  * q05      : out-of-fold 5th-percentile quantile-regression RUL estimate
               (a real conservative lower bound, not rul_hat minus a constant);
  * anom     : a health index in [0,1] = scaled Mahalanobis-style deviation of the
               sensors from the healthy-operation centroid (higher = more degraded).

These are the exact keys the DSL predicates read, so the whole semantic
construction + offline-RL stack runs on real degradation with no other change.
The predicate *meanings* the LLM ensemble scored ("<= 20 cycles left", "anomaly
>= 0.6", ...) transfer directly because C-MAPSS RUL is already in cycles and anom
is in [0,1] -- the SAME frozen plausibility cache therefore serves every subset.

The four C-MAPSS subsets differ in operating conditions and fault modes:

  subset  train engines  operating conditions  fault modes
  FD001   100            1                     1 (HPC)
  FD002   260            6                     1 (HPC)
  FD003   100            1                     2 (HPC + fan)
  FD004   249            6                     2 (HPC + fan)

For the multi-condition subsets (FD002, FD004) the six operating regimes shift
the raw sensors far more than degradation does, which swamps the health signal
under a single global scaler. We therefore cluster (op1,op2,op3) into the six
regimes (KMeans) and standardize each sensor *within* its regime before fitting
the prognostic models and the health index, recovering a regime-invariant
degradation signal. Single-condition subsets (FD001, FD003) take the original
global-standardization path unchanged, so FD001 features are byte-identical to
the previously published cache.

Features are computed once (this is the slow step) and cached to
data/cmapss/features_<subset>.csv.
Build one:  python3 -m saorl.cmapss --subset FD002
Build all:  python3 -m saorl.cmapss --subset all
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cmapss"
RUL_CAP = 125.0  # standard C-MAPSS piecewise-linear RUL cap

# Operating-condition count per subset (drives per-regime normalization).
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SUBSETS = ("FD001", "FD002", "FD003", "FD004")

_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def _raw_train(subset: str) -> Path:
    return DATA_DIR / f"train_{subset}.txt"


def _features_path(subset: str) -> Path:
    return DATA_DIR / f"features_{subset}.csv"


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- download the subset into data/cmapss/")
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    df.columns = _COLS[: df.shape[1]]
    return df


def _capped_rul(df: pd.DataFrame) -> np.ndarray:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return np.minimum(max_cycle - df["cycle"], RUL_CAP).to_numpy(dtype=float)


def _regime_labels(df: pd.DataFrame, n_regimes: int, seed: int) -> np.ndarray:
    """Cluster (op1,op2,op3) into n_regimes operating conditions.

    The C-MAPSS operating settings fall into tight, well-separated clusters, so
    KMeans recovers the documented regimes exactly. A no-op (single regime) for
    the single-condition subsets."""
    if n_regimes <= 1:
        return np.zeros(len(df), dtype=int)
    from sklearn.cluster import KMeans

    ops = df[["op1", "op2", "op3"]].to_numpy(dtype=float)
    return KMeans(n_clusters=n_regimes, n_init=10, random_state=seed).fit_predict(ops)


def _regime_standardize(X: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    """Z-score each column WITHIN each operating regime.

    Makes sensor readings comparable across the six conditions of FD002/FD004 so
    a single degradation signal is recoverable. For one regime this is a plain
    global z-score."""
    Xs = np.empty_like(X, dtype=float)
    for r in np.unique(regimes):
        m = regimes == r
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Xs[m] = (X[m] - mu) / sd
    return Xs


def build_features(subset: str = "FD001", seed: int = 0) -> pd.DataFrame:
    """Train real prognostic models (out-of-fold) and assemble the feature frame.

    For single-condition subsets (FD001/FD003) this is the original global-scaler
    pipeline; for multi-condition subsets (FD002/FD004) sensors are standardized
    per operating regime first (see module docstring)."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    df = load_raw(_raw_train(subset))
    rul = _capped_rul(df)
    n_reg = N_REGIMES.get(subset, 1)

    all_sensors = [c for c in df.columns if c.startswith("s")]
    Xall = df[all_sensors].to_numpy(dtype=float)
    groups = df["unit"].to_numpy()

    if n_reg > 1:
        # Per-regime standardization: removes the operating-condition offset so the
        # informative-sensor filter and the models see degradation, not regime.
        regimes = _regime_labels(df, n_reg, seed)
        Xnorm = _regime_standardize(Xall, regimes)
        keep = Xnorm.std(axis=0) > 1e-6
        Xmodel = Xnorm[:, keep]
        Xs = Xmodel  # health index uses the same regime-normalized sensors
    else:
        # Original FD001 path: raw informative sensors for the models, a single
        # global StandardScaler for the health index. Byte-identical to the cache.
        keep = Xall.std(axis=0) > 1e-6
        Xmodel = Xall[:, keep]
        Xs = StandardScaler().fit_transform(Xmodel)

    gkf = GroupKFold(n_splits=5)

    # out-of-fold RUL point estimate (honest: never predicts an engine it trained on)
    rul_hat = np.zeros(len(df))
    q05 = np.zeros(len(df))
    for tr, te in gkf.split(Xmodel, rul, groups):
        rf = RandomForestRegressor(
            n_estimators=120, max_depth=14, n_jobs=-1, random_state=seed
        )
        rf.fit(Xmodel[tr], rul[tr])
        rul_hat[te] = rf.predict(Xmodel[te])
        gbr = GradientBoostingRegressor(
            loss="quantile", alpha=0.05, n_estimators=80, max_depth=3,
            random_state=seed,
        )
        gbr.fit(Xmodel[tr], rul[tr])
        q05[te] = gbr.predict(Xmodel[te])
    rul_hat = np.clip(rul_hat, 0.0, None)
    q05 = np.clip(np.minimum(q05, rul_hat), 0.0, None)

    # health index: scaled deviation of standardized sensors from healthy centroid
    healthy = rul >= 100.0
    centroid = Xs[healthy].mean(axis=0)
    dist = np.linalg.norm(Xs - centroid, axis=1)
    lo, hi = np.percentile(dist, 1), np.percentile(dist, 99)
    anom = np.clip((dist - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    out = pd.DataFrame(
        dict(unit=df["unit"], cycle=df["cycle"], rul_true=rul,
             rul_hat=rul_hat, q05=q05, anom=anom)
    )
    return out


def load_features(subset: str = "FD001") -> pd.DataFrame:
    path = _features_path(subset)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python3 -m saorl.cmapss --subset {subset}` to build it"
        )
    return pd.read_csv(path)


def to_trajectories(feat: pd.DataFrame) -> List[List[dict]]:
    """One trajectory (list of state dicts) per engine, ordered by cycle."""
    trajs = []
    for _, g in feat.sort_values(["unit", "cycle"]).groupby("unit"):
        trajs.append([
            dict(rul_true=float(r.rul_true), rul_hat=float(r.rul_hat),
                 q05=float(r.q05), anom=float(r.anom))
            for r in g.itertuples()
        ])
    return trajs


def _report(feat: pd.DataFrame) -> None:
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(feat["rul_true"], feat["rul_hat"])
    print(f"  engines={feat['unit'].nunique()}  cycles={len(feat)}  "
          f"RUL MAE(oof)={mae:.1f}")
    band = feat[(feat.rul_true >= 15) & (feat.rul_true <= 25)]
    print(f"  anom: healthy(RUL>=100) median={feat.loc[feat.rul_true>=100,'anom'].median():.2f}"
          f"  incident-band(RUL~20) median={band['anom'].median():.2f}"
          f"  failure(RUL<=5) median={feat.loc[feat.rul_true<=5,'anom'].median():.2f}")
    print(f"  incident cycles(RUL<=20)={int((feat.rul_true<=20).sum())}  "
          f"normal cycles(RUL>=80)={int((feat.rul_true>=80).sum())}")


def _build_one(subset: str) -> None:
    print(f"Building real C-MAPSS {subset} prognostic features "
          f"({N_REGIMES.get(subset, 1)} operating regime(s), out-of-fold)...")
    feat = build_features(subset)
    path = _features_path(subset)
    path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(path, index=False)
    print(f"Wrote {path}")
    _report(feat)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="build C-MAPSS prognostic features")
    ap.add_argument("--subset", default="FD001",
                    help="FD001|FD002|FD003|FD004|all")
    args = ap.parse_args()
    subsets = SUBSETS if args.subset.lower() == "all" else (args.subset,)
    for s in subsets:
        _build_one(s)


if __name__ == "__main__":
    main()
