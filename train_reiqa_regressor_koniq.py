
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import joblib

def split_indices(n, seed, ratios=(0.7, 0.1, 0.2)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

def eval_split(X, y, alpha_grid, seed):
    tr, va, te = split_indices(len(y), seed)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    best_alpha = None
    best_mse = 1e18
    for a in alpha_grid:
        model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                          ("ridge", Ridge(alpha=a, random_state=seed))])
        model.fit(Xtr, ytr)
        pred_va = model.predict(Xva)
        mse = mean_squared_error(yva, pred_va)
        if mse < best_mse:
            best_mse = mse
            best_alpha = a

    # retrain on train+val
    model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                      ("ridge", Ridge(alpha=best_alpha, random_state=seed))])
    model.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))

    pred_te = model.predict(Xte)
    srcc = spearmanr(yte, pred_te).correlation
    plcc = pearsonr(yte, pred_te).statistic
    return best_alpha, srcc, plcc

def main():
    parser = argparse.ArgumentParser(description="Train linear regressor on Re-IQA features for KonIQ")
    parser.add_argument("--features_npz", type=str, default="./koniq_features_reiqa.npz")
    parser.add_argument("--scores_csv", type=str, required=True)
    parser.add_argument("--target", type=str, choices=["MOS", "MOS_zscore"], default="MOS")
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--out_model", type=str, default="./reiqa_ridge_koniq.pkl")
    args = parser.parse_args()

    # Load features
    data = np.load(args.features_npz, allow_pickle=True)
    X = data["X"]
    names = data["names"].astype(str)

    # Load MOS
    df = pd.read_csv(args.scores_csv)
    df = df[["image_name", args.target]].dropna()
    # align
    df_map = {k: v for k, v in zip(df["image_name"].astype(str), df[args.target].astype(float))}
    y = np.array([df_map[n] for n in names if n in df_map])
    X_aligned = np.vstack([X[i] for i, n in enumerate(names) if n in df_map])
    names_aligned = [n for n in names if n in df_map]

    print(f"Aligned samples: {len(names_aligned)} | Feature dim: {X_aligned.shape[1]}")

    # Alpha grid (logspace)
    alpha_grid = np.logspace(-3, 3, 13)

    seeds = [args.seed_base + i for i in range(args.splits)]
    alphas = []
    srccs = []
    plccs = []

    for s in seeds:
        a, sr, pl = eval_split(X_aligned, y, alpha_grid, seed=s)
        alphas.append(a)
        srccs.append(sr)
        plccs.append(pl)
        print(f"[seed={s}] alpha={a:.4f}  SRCC={sr:.4f}  PLCC={pl:.4f}")

    # Aggregate
    srccs_np = np.array(srccs)
    plccs_np = np.array(plccs)
    alpha_counts = Counter(alphas)
    predominant_alpha = max(alpha_counts, key=alpha_counts.get)

    print("---- Summary (10 random splits) ----")
    print(f"SRCC: mean={srccs_np.mean():.4f}  median={np.median(srccs_np):.4f}")
    print(f"PLCC: mean={plccs_np.mean():.4f}  median={np.median(plccs_np):.4f}")
    print(f"Predominant alpha: {predominant_alpha}  (mode over splits)")

    # Final model on ALL data with predominant alpha
    final_model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                            ("ridge", Ridge(alpha=predominant_alpha, random_state=args.seed_base))])
    final_model.fit(X_aligned, y)
    joblib.dump(final_model, args.out_model)
    print(f"Saved final model to: {args.out_model}")

if __name__ == "__main__":
    main()
