import os
import torch
import numpy as np
import random
import torch.nn.functional as F


def save_model(save_path, epoch, model, optimizer):
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_path)


def count_params_and_size(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_mb = total_bytes / (1024 ** 2)

    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    trainable_mb = trainable_bytes / (1024 ** 2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_mb_params_only": total_mb,
        "trainable_mb_params_only": trainable_mb,
    }



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_best_results(results, best_results, epoch, model, optimizer, ckpt_root, seed, save_best_model):
    if epoch == 1:
        for key, value in results.items():
            best_results[key] = value
    else:
        for key, value in results.items():
            if (key == 'Has0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Has0_F1_score'] = results['Has0_F1_score']

                if save_best_model:
                    key_eval = 'Has0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif (key == 'Non0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Non0_F1_score'] = results['Non0_F1_score']

                if save_best_model:
                    key_eval = 'Non0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif key == 'MAE' and value < best_results[key]:
                best_results[key] = value
                # best_results['Corr'] = results['Corr']

                if save_best_model:
                    key_eval = 'MAE'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_2' and (value > best_results[key]):
                best_results[key] = value
                best_results['F1_score'] = results['F1_score']

                if save_best_model:
                    key_eval = 'Mult_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_3' or key == 'Mult_acc_5' or key == 'Mult_acc_7' or key == 'Corr':
                if value > best_results[key]:
                    best_results[key] = value

                    if save_best_model:
                        key_eval = key
                        ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                        save_model(ckpt_path, epoch, model, optimizer)
            
            else:
                pass
    
    return best_results


def dist_to_scalar(
    dist: torch.Tensor,
    K: int = None,
    *,
    support: torch.Tensor = None,
    mode: str = "expectation",
    temperature: float = 1.0,
    keepdim: bool = True,
    normalize: bool = True,
    input_type: str = "auto",   # "auto" | "prob" | "logits"
    eps: float = 1e-8,
):
    assert dist.dim() == 2, f"dist must be [B,K], got {dist.shape}"
    B, K_infer = dist.shape
    K = K or K_infer
    assert K == K_infer, f"K mismatch: K={K}, dist.shape[-1]={K_infer}"

    device, dtype = dist.device, dist.dtype

    # ---- build support ----
    if support is None:
        if K % 2 == 1:
            mid = K // 2
            support = torch.arange(-mid, mid + 1, device=device, dtype=dtype)
        else:
            mid = K / 2
            support = (torch.arange(K, device=device, dtype=dtype) - (mid - 0.5))
    else:
        support = support.to(device=device, dtype=dtype)
        assert support.numel() == K and support.dim() == 1

    # ---- normalize ----
    if not normalize:
        p = dist
        logits = None
    else:
        if input_type not in ("auto", "prob", "logits"):
            raise ValueError(f"input_type must be auto/prob/logits, got {input_type}")

        if input_type == "logits":
            logits = dist
            p = F.softmax(logits, dim=-1)
        elif input_type == "prob":
            logits = None
            p = dist.clamp_min(0.0)
            p = p / (p.sum(dim=-1, keepdim=True) + eps)
        else:

            if (dist.min() >= -1e-6) and (dist.sum(dim=-1).min() > 0):
                logits = None
                p = dist.clamp_min(0.0)
                p = p / (p.sum(dim=-1, keepdim=True) + eps)
            else:
                logits = dist
                p = F.softmax(logits, dim=-1)

    if mode == "expectation":
        y = (p * support.view(1, -1)).sum(dim=-1, keepdim=keepdim)

    elif mode == "argmax":
        idx = torch.argmax(p, dim=-1)
        y = support[idx]
        if keepdim:
            y = y.view(-1, 1)

    elif mode == "softargmax":
        T = max(float(temperature), eps)
        if logits is not None:
            w = F.softmax(logits / T, dim=-1)
        else:
            w = (p.clamp_min(eps) ** (1.0 / T))
            w = w / (w.sum(dim=-1, keepdim=True) + eps)
        y = (w * support.view(1, -1)).sum(dim=-1, keepdim=keepdim)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return y, p, support
