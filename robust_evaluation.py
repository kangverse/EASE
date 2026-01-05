
import os
import json
import torch
import yaml
import argparse
from core.dataset import MMDataEvaluationLoader
from models.EASE import build_model
from core.metric import MetricsTop

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--file_package', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--config_file', type=str, default='')
parser.add_argument('--key_eval', type=str, default='')
opt = parser.parse_args()
print(opt)

def to_jsonable(x):
    import numpy as np
    import torch

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):   # np.float32 / np.float64
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    return x

def compute_avg_std(list_of_dicts):
    keys = sorted(set().union(*[d.keys() for d in list_of_dicts]))
    avg, std = {}, {}
    for k in keys:
        vals = [float(d[k]) for d in list_of_dicts if k in d]
        if len(vals) == 0:
            continue
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(len(vals), 1)
        avg[k] = m
        std[k] = v ** 0.5
    return avg, std



def metric_summary_str(dataset_name, key_eval, missing_rate, avg_dict):
    metric_order = [
        "Has0_acc_2", "Non0_acc_2",
        "Has0_F1_score", "Non0_F1_score",
        "Mult_acc_5", "Mult_acc_7",
        "MAE", "Corr",
    ]

    ignore_keys = {"_seed"}

    parts = []
    used = set()
    for k in metric_order:
        if k in avg_dict and k not in ignore_keys:
            parts.append(f"{k}={avg_dict[k]:.4f}")
            used.add(k)


    rest_keys = sorted([k for k in avg_dict.keys() if k not in used and k not in ignore_keys])
    for k in rest_keys:
        parts.append(f"{k}={avg_dict[k]:.4f}")

    if not parts:
        return f"{dataset_name} | missing_rate={missing_rate} | AVG N/A"

    return f"{dataset_name} | missing_rate={missing_rate} | " + " | ".join(parts)



def evaluate(model, eval_loader, metrics):
    y_pred, y_true = [], []

    model.eval()
    for _, data in enumerate(eval_loader):
        incomplete_input = (
            data['vision_m'].to(device),
            data['audio_m'].to(device),
            data['text_m'].to(device)
        )
        sentiment_labels = data['labels']['M'].to(device)

        with torch.no_grad():
            out = model((None, None, None), incomplete_input)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(sentiment_labels.cpu())

    pred = torch.cat(y_pred)
    true = torch.cat(y_true)
    return metrics(pred, true)


def main():
    config_file = opt.config_file if opt.config_file else 'configs/eval_mosi.yaml'
    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = args['dataset']['datasetName']
    key_eval = opt.key_eval if opt.key_eval else args['base'].get('key_eval', '')

    model = build_model(args).to(device)
    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(dataset_name)

    missing_rate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seeds = [1111, 1112, 1113]

    pkg = opt.file_package
    last = pkg.rstrip("/").split("/")[-1]

    out_dir = f"{opt.save_path}/{last}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_{key_eval}.txt")


    with open(out_path, "w", encoding="utf-8", buffering=1) as fw:
        print(f"Dataset: {dataset_name}", file=fw, flush=True)
        print(f"Key Eval: {key_eval}", file=fw, flush=True)
        print(f"Seeds: {seeds}", file=fw, flush=True)
        print("=" * 60, file=fw, flush=True)

        for cur_r in missing_rate_list:
            per_seed_results = []

            print(f"\n--- missing_rate = {cur_r} ---", file=fw, flush=True)

            for cur_seed in seeds:
                ckpt_path = f'../{opt.file_package}/{dataset_name}/best_{key_eval}_{cur_seed}.pth'
                print(f"[LOAD] seed={cur_seed} ckpt={ckpt_path}", file=fw, flush=True)

                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt['state_dict'])

                args['base']['missing_rate_eval_test'] = cur_r
                loader = MMDataEvaluationLoader(args)

                results = evaluate(model, loader, metrics)
                results['_seed'] = int(cur_seed)
                results = to_jsonable(results)
                per_seed_results.append(results)

                print("[PER_SEED_ONE] " + json.dumps(results, ensure_ascii=False), file=fw, flush=True)

            avg_dict, std_dict = compute_avg_std(per_seed_results)

            print("[AVG_ALL]", file=fw, flush=True)
            for k in sorted(avg_dict.keys()):
                print(f"  {k}: {avg_dict[k]:.4f}", file=fw, flush=True)

            print("[STD_ALL]", file=fw, flush=True)
            for k in sorted(std_dict.keys()):
                print(f"  {k}: {std_dict[k]:.4f}", file=fw, flush=True)

            print("[SUMMARY_LINE] " + metric_summary_str(dataset_name, key_eval, cur_r, avg_dict),
                  file=fw, flush=True)

            print("-" * 60, file=fw, flush=True)

    print(f"[DONE] Robust evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
