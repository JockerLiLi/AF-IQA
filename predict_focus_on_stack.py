# predict_focus_on_stack.py
import os, argparse, numpy as np, joblib, re
import pandas as pd

def load_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["names"]

def name_to_z(name):
    m = re.search(r"z([+-]?\d+)", name)  # 兼容 z-3 / z3 / z+3
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--features_npz", required=True, help="该 z-stack 的特征（由你的提特征脚本得到）")
    args = ap.parse_args()

    model = joblib.load(args.model_path)
    X, names = load_features(args.features_npz)

    yhat = model.predict(X)  # 预测分数，越大越清晰
    z_vals = [name_to_z(n) for n in names]
    df = pd.DataFrame({"name": names, "z": z_vals, "score": yhat})
    df.sort_values(by="score", ascending=False, inplace=True)

    best = df.iloc[0]
    print("\n=== 该 z-stack 评分（前10） ===")
    print(df.head(10).to_string(index=False))
    print(f"\n[RESULT] 最佳帧: {best['name']} | z={best['z']} | 预测分数={best['score']:.6f}")

if __name__ == "__main__":
    main()