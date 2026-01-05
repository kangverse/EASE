import os
import torch
import yaml
import argparse

from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results, count_params_and_size
from core.metric import MetricsTop
import time

from models.EASE import build_model
from core.loss import EASELoss

EXP_ROOT = os.environ.get("EXP_ROOT", "exp")

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='')
parser.add_argument('--seed', type=int, default=-1)
opt = parser.parse_args()
print(opt)




def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (
            data['vision'].to(device),
            data['audio'].to(device),
            data['text'].to(device),
        )
        incomplete_input = (
            data['vision_m'].to(device),
            data['audio_m'].to(device),
            data['text_m'].to(device),
        )

        sentiment_labels = data['labels']['M'].to(device)

        missing_rate_l = data['labels']['missing_rate_l'].to(device)
        missing_rate_a = data['labels']['missing_rate_a'].to(device)
        missing_rate_v = data['labels']['missing_rate_v'].to(device)

        label = {
            'sentiment_labels': sentiment_labels,
            'missing_rate_l': missing_rate_l,
            'missing_rate_a': missing_rate_a,
            'missing_rate_v': missing_rate_v,
        }

        missing_rates = {'l': missing_rate_l, 'a': missing_rate_a, 'v': missing_rate_v}

        out = model(complete_input, incomplete_input, missing_rates=missing_rates)

        loss = loss_fn(out, label)

        optimizer.zero_grad(set_to_none=True)
        loss['loss'].backward()

        optimizer.step()

        y_pred.append(out['sentiment_preds'].detach().cpu())
        y_true.append(sentiment_labels.detach().cpu())

        if cur_iter == 0:
            for k, v in loss.items():
                loss_dict[k] = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
        else:
            for k, v in loss.items():
                loss_dict[k] += float(v.detach().cpu()) if torch.is_tensor(v) else float(v)

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {k: v / (cur_iter + 1) for k, v in loss_dict.items()}
    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')


def evaluate(model, eval_loader, loss_fn, epoch, metrics,
             compute_loss=False, use_complete_for_loss=False):
    y_pred, y_true = [], []
    loss_dict = {}

    model.eval()
    for cur_iter, data in enumerate(eval_loader):
        if use_complete_for_loss:
            complete_input = (
                data['vision'].to(device),
                data['audio'].to(device),
                data['text'].to(device),
            )
        else:
            complete_input = (None, None, None)

        incomplete_input = (
            data['vision_m'].to(device),
            data['audio_m'].to(device),
            data['text_m'].to(device),
        )

        sentiment_labels = data['labels']['M'].to(device)

        missing_rate_l = data['labels']['missing_rate_l'].to(device)
        missing_rate_a = data['labels']['missing_rate_a'].to(device)
        missing_rate_v = data['labels']['missing_rate_v'].to(device)

        label = {
            'sentiment_labels': sentiment_labels,
            'missing_rate_l': missing_rate_l,
            'missing_rate_a': missing_rate_a,
            'missing_rate_v': missing_rate_v,
        }

        missing_rates = {'l': missing_rate_l, 'a': missing_rate_a, 'v': missing_rate_v}

        with torch.no_grad():
            out = model(complete_input, incomplete_input, missing_rates=missing_rates)

            if compute_loss:
                loss = loss_fn(out, label)
                if cur_iter == 0:
                    for k, v in loss.items():
                        loss_dict[k] = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                else:
                    for k, v in loss.items():
                        loss_dict[k] += float(v.detach().cpu()) if torch.is_tensor(v) else float(v)

        y_pred.append(out['sentiment_preds'].detach().cpu())
        y_true.append(sentiment_labels.detach().cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    if compute_loss:
        loss_dict = {k: v / (cur_iter + 1) for k, v in loss_dict.items()}
        print(f'Eval Loss Epoch {epoch}: {loss_dict}')
        print(f'Eval Results Epoch {epoch}: {results}')

    return results


def main():
    best_valid_results, best_test_results = {}, {}

    config_file = 'configs/train_sims.yaml' if opt.config_file == '' else opt.config_file
    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    ckpt_root = os.path.join(
        EXP_ROOT,
        "ckpt",
        args['dataset']['datasetName']
    )
    os.makedirs(ckpt_root, exist_ok=True)
    print("ckpt root :", ckpt_root)

    # model
    model = build_model(args).to(device)
        

    # dataloader
    dataLoader = MMDataLoader(args)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args['base']['lr'],
        weight_decay=args['base']['weight_decay']
    )

    # scheduler
    scheduler_warmup = get_scheduler(optimizer, args)

    # loss
    loss_fn = EASELoss(args).to(device)

    # metrics
    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    # flags
    do_validation = bool(args['base'].get('do_validation', True))
    n_epochs = int(args['base']['n_epochs'])
    
    total_train_start = time.time()

    # main loop
    for epoch in range(1, n_epochs + 1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)

        if do_validation:
            valid_results = evaluate(
                model, dataLoader['valid'], loss_fn, epoch, metrics,
                compute_loss=False,            
                use_complete_for_loss=False   
            )
            best_valid_results = get_best_results(
                valid_results, best_valid_results, epoch,
                model, optimizer, ckpt_root, seed,
                save_best_model=False
            )
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(
            model, dataLoader['test'], loss_fn, epoch, metrics,
            compute_loss=False,
            use_complete_for_loss=False
        )
        best_test_results = get_best_results(
            test_results, best_test_results, epoch,
            model, optimizer, ckpt_root, seed,
            save_best_model=True
        )
        print(f'Current Best Test Results: {best_test_results}\n')

        scheduler_warmup.step()
    
    total_time = time.time() - total_train_start
    print(f"[Time] Total Training Time: {total_time/3600:.2f} hours ({total_time/60:.2f} min)")


if __name__ == '__main__':
    main()
