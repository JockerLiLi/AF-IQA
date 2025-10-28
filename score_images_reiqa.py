# score_images_reiqa.py
import argparse, os, sys, pathlib, glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import joblib
import csv

# ---- 避免 TrainOptions 误解析你自定义的命令行参数 ----
def build_model_and_encoder(ckpt_path, device):
    from options.train_options import TrainOptions
    from networks.build_backbone import build_model

    argv_bak = sys.argv[:]
    sys.argv = [sys.argv[0]]
    args = TrainOptions().parse()
    sys.argv = argv_bak

    # 与权重匹配：MLP 头
    args.head = 'mlp'
    args.device = device

    model, _ = build_model(args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    model.eval()
    return model

def list_images(path):
    exts = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')
    if os.path.isdir(path):
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(path, f'**/*{ext}'), recursive=True)
        return sorted(files)
    elif os.path.isfile(path) and path.lower().endswith(exts):
        return [path]
    else:
        raise FileNotFoundError(f'No images found at: {path}')

def main():
    p = argparse.ArgumentParser(description='Score images with Re-IQA + trained Ridge head')
    p.add_argument('--input', required=True, help='image file or folder')
    p.add_argument('--content_ckpt', default='./re-iqa_ckpts/content_aware_r50.pth')
    p.add_argument('--quality_ckpt', default='./re-iqa_ckpts/quality_aware_r50.pth')
    p.add_argument('--regressor_pkl', default='./reiqa_ridge_koniq.pkl')
    p.add_argument('--out_csv', default='./pred_mos.csv')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    # 设备
    device = torch.device(args.device)

    # 加载两套 encoder
    content_model = build_model_and_encoder(args.content_ckpt, device)
    quality_model = build_model_and_encoder(args.quality_ckpt, device)

    # 归一化器（content 分支用）
    normalize_imagenet = transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])

    # 加载回归头
    reg = joblib.load(args.regressor_pkl)

    # 收集图片
    images = list_images(args.input)
    if len(images) == 0:
        print('No images found.')
        return

    results = []
    for idx, path in enumerate(images, 1):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f'[WARN] Skip {path}: {e}')
            continue

        half = img.resize((max(1,img.size[0]//2), max(1,img.size[1]//2)))

        t_full = transforms.ToTensor()(img).unsqueeze(0).to(device)
        t_half = transforms.ToTensor()(half).unsqueeze(0).to(device)

        # content-aware features（用 ImageNet 归一化）
        with torch.no_grad():
            t_full_c = normalize_imagenet(t_full.squeeze(0)).unsqueeze(0)
            t_half_c = normalize_imagenet(t_half.squeeze(0)).unsqueeze(0)
            f1_c = content_model.module.encoder(t_full_c)
            f2_c = content_model.module.encoder(t_half_c)
            feat_c = torch.cat((f1_c, f2_c), dim=1)

        # quality-aware features（不做归一化）
        with torch.no_grad():
            f1_q = quality_model.module.encoder(t_full)
            f2_q = quality_model.module.encoder(t_half)
            feat_q = torch.cat((f1_q, f2_q), dim=1)

        feat = torch.cat((feat_c, feat_q), dim=1).detach().cpu().numpy().reshape(1,-1)

        mos = float(reg.predict(feat)[0])  # 预测 MOS（KonIQ 的标尺）
        results.append((path, mos))
        if idx <= 5:
            print(f'[#{idx:04d}] {path}  MOS={mos:.3f}')

    # 写出 CSV
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_path','pred_mos'])
        for r in results:
            w.writerow(r)

    print(f'Wrote {len(results)} predictions to: {args.out_csv}')

if __name__ == '__main__':
    main()
