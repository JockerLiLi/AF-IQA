
import argparse
import os
import ntpath
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Import repo modules (assumes you run from repo root)
from options.train_options import TrainOptions
from networks.build_backbone import build_model

def load_encoder(ckpt_path, device, normalize=False):
    import sys
    # 关键：避免 TrainOptions 误解析自定义参数
    argv_bak = sys.argv[:]
    sys.argv = [sys.argv[0]]

    # 用默认参数解析
    args = TrainOptions().parse()
    sys.argv = argv_bak

    # === 新增：强制使用与权重匹配的 MLP 头 ===
    args.head = 'mlp'
    # （可选）确保架构一致；默认就是 resnet50，一般不用改
    # args.arch = 'resnet50'
    # =======================================

    args.device = device
    model, _ = build_model(args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Extract Re-IQA features for KonIQ images")
    parser.add_argument("--csv_images", type=str, required=True, help="Path to koniq_images.csv (has one column Image_path)")
    parser.add_argument("--dataset_root", type=str, default=None, help="If provided, we will rebuild paths as dataset_root / basename(Image_path)")
    parser.add_argument("--content_ckpt", type=str, default="./re-iqa_ckpts/content_aware_r50.pth")
    parser.add_argument("--quality_ckpt", type=str, default="./re-iqa_ckpts/quality_aware_r50.pth")
    parser.add_argument("--out_npz", type=str, default="./koniq_features_reiqa.npz")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv_images)
    # Build absolute paths robustly
    names = []
    paths = []
    for p in df["Image_path"].astype(str).tolist():
        name = ntpath.basename(p)
        names.append(name)
        if args.dataset_root:
            paths.append(os.path.join(args.dataset_root, name))
        else:
            paths.append(p)

    # Encoders
    device = torch.device(args.device)
    content_model = load_encoder(args.content_ckpt, device, normalize=True)
    quality_model = load_encoder(args.quality_ckpt, device, normalize=False)

    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    feats = []
    bad = []
    for i, (name, path) in enumerate(zip(names, paths)):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skip {name}: {e}")
            bad.append(name)
            continue

        # original + half-scale
        image_half = image.resize((image.size[0]//2, image.size[1]//2))

        # Tensors
        t_full = transforms.ToTensor()(image).unsqueeze(0).to(device)
        t_half = transforms.ToTensor()(image_half).unsqueeze(0).to(device)

        # Content-aware (with ImageNet normalization)
        with torch.no_grad():
            t_full_c = normalize_imagenet(t_full.squeeze(0)).unsqueeze(0)
            t_half_c = normalize_imagenet(t_half.squeeze(0)).unsqueeze(0)
            f1_c = content_model.module.encoder(t_full_c)
            f2_c = content_model.module.encoder(t_half_c)
            feat_c = torch.cat((f1_c, f2_c), dim=1)

        # Quality-aware (no normalization per demo)
        with torch.no_grad():
            f1_q = quality_model.module.encoder(t_full)
            f2_q = quality_model.module.encoder(t_half)
            feat_q = torch.cat((f1_q, f2_q), dim=1)

        feat = torch.cat((feat_c, feat_q), dim=1).detach().cpu().numpy().squeeze()
        feats.append(feat)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(names)}] processed")

    X = np.vstack(feats) if len(feats) else np.empty((0,))
    names_ok = [n for n in names if n not in bad]

    np.savez_compressed(args.out_npz, X=X, names=np.array(names_ok))
    print(f"Saved features: {args.out_npz} | N={len(names_ok)} | D={X.shape[1] if X.size else 0}")
    if bad:
        print(f"Skipped {len(bad)} images (could not open).")

if __name__ == "__main__":
    main()
