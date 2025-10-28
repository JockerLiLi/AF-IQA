# train_focus_regressor.py
import os, re, argparse, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
import joblib

def extract_basename(p):
    return os.path.basename(p)

def extract_stack_id(p):
    # 期望路径中含 ...\S001\... 或 .../S001/...
    m = re.search(r"[\\/](S\d{3,})[\\/]", p)
    return m.group(1) if m else "UNK"

def score_from_z(z, sigma=2.0):
    # 改进型“高斯风格”分数：1 / (1 + (|z|/sigma)^2)
    return 1.0 / (1.0 + (abs(float(z)) / float(sigma))**2)

def align_features_labels(npz_path, csv_path, target_col="focus_score", fallback_from_z=True, sigma=2.0):
    # 加载特征
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                  # [N, 8192]
    names = data["names"]          # [N] basenames

    # 加载标签 CSV
    df = pd.read_csv(csv_path)

    # 取 basename 以便 join
    if "image_path" in df.columns:
        df["basename"] = df["image_path"].apply(extract_basename)
        df["stack_id"] = df["image_path"].apply(extract_stack_id)
    else:
        # 如果第一列就是路径
        first_col = df.columns[0]
        df["basename"] = df[first_col].astype(str).apply(extract_basename)
        df["stack_id"] = df[first_col].astype(str).apply(extract_stack_id)

    # 兜底：若没有 focus_score，则尝试从 z_um 生成
    if target_col not in df.columns:
        if fallback_from_z and ("z_um" in df.columns):
            warnings.warn(f"[WARN] 未找到列 `{target_col}`；从 z_um 生成分数（sigma={sigma}）。")
            df[target_col] = df["z_um"].apply(lambda z: score_from_z(z, sigma=sigma))
        else:
            raise ValueError(f"CSV 中既无 `{target_col}`，也无 `z_um`，无法构造监督信号。")

    # 建立索引映射：basename -> 行
    df_map = df.set_index("basename")
    rows, groups, missed = [], [], []

    for i, nm in enumerate(names):
        if nm in df_map.index:
            rows.append(df_map.loc[nm])
            groups.append(df_map.loc[nm]["stack_id"])
        else:
            missed.append(nm)

    if len(rows) == 0:
        raise RuntimeError("一个样本都没对齐上。请检查 mic_features.npz 的 names 与 CSV 的 image_path/basename 是否一致。")

    if missed:
        print(f"[WARN] 有 {len(missed)} 个特征未在 CSV 对应到标签（将被跳过）。例如：{missed[:5]}")

    Y = np.array([r[target_col] for r in rows], dtype=np.float32)
    G = np.array(groups)
    # 只保留能对齐的 X
    mask_keep = np.isin(names, list(df_map.index))
    X_keep = X[mask_keep]

    # 基本 sanity check
    if X_keep.shape[0] != Y.shape[0]:
        raise RuntimeError(f"对齐后的样本数不一致：X={X_keep.shape[0]} vs Y={Y.shape[0]}")

    return X_keep, Y, G, names[mask_keep]

def evaluate_loso(X, Y, G, alpha=1000.0, n_splits=None):
    # GroupKFold 按 stack_id（G）分组，相当于 LOSO
    unique_groups = np.unique(G)
    n_splits = len(unique_groups) if n_splits is None else n_splits
    gkf = GroupKFold(n_splits=n_splits)

    srccs, plccs = [], []
    for tr_idx, te_idx in gkf.split(X, Y, groups=G):
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                              Ridge(alpha=alpha, random_state=0))
        model.fit(X[tr_idx], Y[tr_idx])
        y_hat = model.predict(X[te_idx])

        srcc = spearmanr(Y[te_idx], y_hat).correlation
        plcc = pearsonr(Y[te_idx], y_hat)[0]
        srccs.append(srcc); plccs.append(plcc)

    return float(np.nanmean(srccs)), float(np.nanmean(plccs))

def train_full_and_save(X, Y, alpha, out_path):
    model = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                          Ridge(alpha=alpha, random_state=0))
    model.fit(X, Y)
    joblib.dump(model, out_path)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_npz", required=True)
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--target", default="focus_score")
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--sigma", type=float, default=2.0, help="当从 z_um 构造分数时使用")
    ap.add_argument("--out_model", required=True)
    args = ap.parse_args()

    X, Y, G, kept_names = align_features_labels(
        args.features_npz, args.scores_csv, target_col=args.target, fallback_from_z=True, sigma=args.sigma
    )
    print(f"[INFO] 对齐样本: {len(Y)} | 特征维度: {X.shape[1]} | stack 组数: {len(np.unique(G))}")

    srcc, plcc = evaluate_loso(X, Y, G, alpha=args.alpha)
    print(f"[EVAL][LOSO]  Ridge alpha={args.alpha:.1f} | SRCC={srcc:.4f} | PLCC={plcc:.4f}")

    model = train_full_and_save(X, Y, alpha=args.alpha, out_path=args.out_model)
    print(f"[DONE] 已保存模型到: {args.out_model}")

if __name__ == "__main__":
    main()
