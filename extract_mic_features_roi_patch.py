# extract_mic_features_roi_patch.py
# 可直接运行的无-global 版本（PowerShell/Linux/Mac均可）
import os, csv, argparse, random, math, ntpath
import numpy as np
import tifffile as tiff
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

# ====== 复用你仓库里的加载函数（保持一致）======
LOAD_FROM_REPO = True
try:
    from extract_reiqa_features_koniq import load_encoder as _repo_load_encoder
except Exception:
    LOAD_FROM_REPO = False
    print("[WARN] 未找到 extract_reiqa_features_koniq.load_encoder；"
          "请把本脚本放在你的 Re-IQA 工程根目录或把该函数所在文件加入 PYTHONPATH。")

# ====== 常量（仅作默认值展示，不会被修改）======
DEFAULT_PATCH_SIZE      = 256
DEFAULT_PATCHES_PER_IMG = 32
DEFAULT_TOPK_FRAC       = 0.20
DEFAULT_MIN_ROI_COVER   = 0.40
DEFAULT_P_LOW           = 1.0
DEFAULT_P_HIGH          = 99.8
ROI_MIN_AREA            = 64 * 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- 工具函数 ----------

def read_image_float01(path, p_low, p_high):
    """读取TIF并按分位数标准化到[0,1]；兼容单/多通道，自动转灰度"""
    img = tiff.imread(path)  # (H,W) 或 (H,W,C)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)

    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        lo, hi = img.min(), img.max()
        if hi <= lo:  # 极端兜底
            hi = lo + 1.0
    x = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return x  # (H,W) in [0,1]

def make_roi_mask(img01):
    """Otsu 阈值 + 开闭运算 + 小连通域过滤；失败时放宽为全图"""
    u8 = (img01 * 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= ROI_MIN_AREA:
            out[labels == i] = 255
    if out.sum() == 0:
        out[:] = 255
    return (out > 0).astype(np.uint8)

def sample_patch_positions(mask, patch_size, n, min_cover=0.4, max_trials=10000):
    """在 ROI 内随机采样 patch 左上角坐标；不足时用网格补齐，再不足放宽全图"""
    H, W = mask.shape
    ps = patch_size
    cand, trials = [], 0
    ys = range(0, H-ps+1)
    xs = range(0, W-ps+1)
    while len(cand) < n and trials < max_trials:
        y = random.choice(ys); x = random.choice(xs)
        if mask[y:y+ps, x:x+ps].mean() >= min_cover:
            cand.append((y, x))
        trials += 1
    if len(cand) < n:
        step = max(1, (min(H, W) - ps) // int(math.sqrt(n) + 1))
        for y in range(0, H-ps+1, step):
            for x in range(0, W-ps+1, step):
                if mask[y:y+ps, x:x+ps].mean() >= min_cover:
                    cand.append((y, x))
                if len(cand) >= n: break
            if len(cand) >= n: break
    if len(cand) == 0:
        for _ in range(n):
            y = random.randint(0, H-ps); x = random.randint(0, W-ps)
            cand.append((y, x))
    return cand[:n]

def laplacian_var(patch01):
    """拉普拉斯方差作为纹理强度评分"""
    g = (patch01 * 255).astype(np.uint8)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

to_tensor = transforms.ToTensor()
imagenet_norm = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std =[0.229,0.224,0.225])

def prep_for_content(patch01):
    rgb = np.repeat(patch01[..., None], 3, axis=2)
    t = to_tensor(rgb)
    t = imagenet_norm(t)
    return t

def prep_for_quality(patch01):
    rgb = np.repeat(patch01[..., None], 3, axis=2)
    t = to_tensor(rgb)
    return t

def load_encoder(ckpt_path, normalize_for_content=True):
    """复用你工程里的权重加载逻辑"""
    if not LOAD_FROM_REPO:
        raise RuntimeError("未能导入仓库的 load_encoder。请确认脚本位置或 PYTHONPATH。")
    model = _repo_load_encoder(ckpt_path, device, normalize=normalize_for_content)
    model.eval()
    return model

def forward_encoder(model, x4d):
    with torch.no_grad():
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "encoder"):
            f = m.encoder(x4d)
        else:
            f = m(x4d)
        if f.ndim == 4:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f  # [B, 2048]

def extract_feat_for_one_image(path, content_model, quality_model,
                               patch_size, n_patches, topk_frac,
                               p_low, p_high, min_roi_cover):
    """单帧：ROI→随机patch→双尺度×2分支→拼接8192→按纹理Top-k均值"""
    img01 = read_image_float01(path, p_low, p_high)
    mask  = make_roi_mask(img01)
    pos   = sample_patch_positions(mask, patch_size, n_patches, min_cover=min_roi_cover)

    feats, scores = [], []
    for (y, x) in pos:
        p = img01[y:y+patch_size, x:x+patch_size]
        s = laplacian_var(p); scores.append(s)
        p_half = cv2.resize(p, (patch_size//2, patch_size//2), interpolation=cv2.INTER_AREA)

        tc1 = prep_for_content(p).unsqueeze(0).to(device)
        tc2 = prep_for_content(p_half).unsqueeze(0).to(device)
        tq1 = prep_for_quality(p).unsqueeze(0).to(device)
        tq2 = prep_for_quality(p_half).unsqueeze(0).to(device)

        f1c = forward_encoder(content_model, tc1)
        f2c = forward_encoder(content_model, tc2)
        f1q = forward_encoder(quality_model, tq1)
        f2q = forward_encoder(quality_model, tq2)

        f = torch.cat([f1c, f2c, f1q, f2q], dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        feats.append(f)

    feats  = np.stack(feats, axis=0)     # [P,8192]
    scores = np.asarray(scores)          # [P]
    k = max(1, int(math.ceil(len(scores) * topk_frac)))
    top_idx = np.argsort(-scores)[:k]
    feat_img = feats[top_idx].mean(axis=0)  # [8192]
    return feat_img

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_images", required=True, help="CSV，含一列 image_path")
    ap.add_argument("--content_ckpt", required=True)
    ap.add_argument("--quality_ckpt", required=True)
    ap.add_argument("--out_npz", required=True)

    ap.add_argument("--patch_size",      type=int,   default=DEFAULT_PATCH_SIZE)
    ap.add_argument("--patches_per_img", type=int,   default=DEFAULT_PATCHES_PER_IMG)
    ap.add_argument("--topk_frac",       type=float, default=DEFAULT_TOPK_FRAC)
    ap.add_argument("--min_roi_cover",   type=float, default=DEFAULT_MIN_ROI_COVER)
    ap.add_argument("--p_low",           type=float, default=DEFAULT_P_LOW)
    ap.add_argument("--p_high",          type=float, default=DEFAULT_P_HIGH)
    ap.add_argument("--seed",            type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    content_model = load_encoder(args.content_ckpt, normalize_for_content=True)
    quality_model = load_encoder(args.quality_ckpt, normalize_for_content=False)

    # 读取 CSV（允许列名不是 image_path，则取第一列作为路径）
    img_paths = []
    with open(args.csv_images, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        col = "image_path" if "image_path" in r.fieldnames else r.fieldnames[0]
        for t in r:
            p = t[col]
            if os.path.isfile(p):
                img_paths.append(p)
            else:
                print(f"[WARN] 跳过不存在的路径: {p}")

    X, names = [], []
    for i, p in enumerate(img_paths):
        try:
            feat = extract_feat_for_one_image(
                p, content_model, quality_model,
                patch_size=args.patch_size,
                n_patches=args.patches_per_img,
                topk_frac=args.topk_frac,
                p_low=args.p_low, p_high=args.p_high,
                min_roi_cover=args.min_roi_cover,
            )
            X.append(feat); names.append(ntpath.basename(p))
        except Exception as e:
            print(f"[ERR] 提特征失败 {p}: {e}")

        if (i + 1) % 20 == 0:
            print(f"[INFO] 已完成 {i+1}/{len(img_paths)}")

    if len(X) == 0:
        raise RuntimeError("没有成功提取到任何特征，请检查 CSV 路径列与图像文件是否存在。")

    X = np.stack(X, axis=0).astype(np.float32)
    names = np.asarray(names)
    np.savez(args.out_npz, X=X, names=names)
    print(f"[DONE] 保存到 {args.out_npz} | N={len(names)} | dim={X.shape[1]}")

if __name__ == "__main__":
    main()
