
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from .basic_layers import Transformer
from .bert import BertTextEncoder



class RouterMLP(nn.Module):
    """Token-wise router: [B,T,D] -> [B,T,K] softmax weights"""
    def __init__(self, dim: int, hidden: int = None, out_k: int = 2, drop: float = 0.0):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_k),
        )

    def forward(self, x):
        # x: [B,T,D]
        logits = self.net(x)                 # [B,T,K]
        w = torch.softmax(logits, dim=-1)    # [B,T,K]
        return w, logits



class SemanticMappingBackbone(nn.Module):
    """
    Transformer-based Adapter backbone:
      h_shared = FFN(MHSA(z_src))
    这里用你们现成的 Transformer 实现 MHSA+FFN 的组合
    """
    def __init__(self, dim=128, depth=2, heads=8, mlp_dim=128, num_frames=24):
        super().__init__()
        self.net = Transformer(
            num_frames=num_frames,
            save_hidden=False,
            token_len=None,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )

    def forward(self, z_src):
        # z_src: [B, Nsrc, D]
        return self.net(z_src)  # [B, Nsrc, D]


class EvidentialDualHead(nn.Module):
    """
    Dual-head predictor:
      mu = H_content(h_shared)
      sigma2 = softplus(H_uncert(h_shared)) + eps
    """
    def __init__(self, dim=128, hidden=128, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.content = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.uncert = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, h_shared):
        mu = self.content(h_shared)
        sigma2 = F.softplus(self.uncert(h_shared)) + self.eps
        return mu, sigma2


class UCMIModule(nn.Module):
    """
    ucmi: z_src -> h_shared -> (mu, sigma2) -> z_hat ~ N(mu, diag(sigma2))
    """
    def __init__(self, dim=128, depth=2, heads=8, mlp_dim=128, eps=1e-6, num_frames=24):
        super().__init__()
        self.backbone = SemanticMappingBackbone(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, num_frames=num_frames)
        self.dual_head = EvidentialDualHead(dim=dim, hidden=mlp_dim, eps=eps)

    def forward(self, z_src, out_tokens=8, sample=True):
        """
        z_src: [B, Nsrc, D]
        return:
          mu, sigma2, z_hat: [B, out_tokens, D]
          u_token: [B, out_tokens]  (per-token uncertainty)
          u: [B, 1]                 (sample-level uncertainty)
        """
        h = self.backbone(z_src)[:, :out_tokens]          
        mu, sigma2 = self.dual_head(h)                    # [B, T, D]
        if sample:
            eps = torch.randn_like(mu)
            z_hat = mu + torch.sqrt(sigma2) * eps
        else:
            z_hat = mu

        u_token = sigma2.mean(dim=-1)                     # [B, T]
        u = u_token.mean(dim=1, keepdim=True)             # [B, 1]
        return mu, sigma2, z_hat, u_token, u


class CompletenessEstimator(nn.Module):
    def __init__(self, dim=128, depth=2, heads=8, mlp_dim=128, token_len=8):
        super().__init__()
        self.encoder = Transformer(
            num_frames=token_len,
            save_hidden=False,
            token_len=1,   # 输出一个 summary token
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, tok):
        # tok: [B, T, D]
        s = self.encoder(tok)[:, :1].squeeze(1)  # [B, D]
        w = self.head(s)                         # [B, 1]
        return w


def estimate_text_missing_rate(language_m):
    """
    language_m: [B, 3, L]  (input_ids, input_mask, segment_ids)
    UNK token is 100 in your dataloader.
    return: missing_rate in [0,1], shape [B,1]
    """
    input_ids = language_m[:, 0, :]   # [B, L]
    input_mask = language_m[:, 1, :]  # [B, L]
    valid = (input_mask > 0.5)
    miss = (input_ids == 100) & valid
    denom = valid.sum(dim=1).clamp_min(1.0)
    rate = miss.sum(dim=1).float() / denom.float()
    return rate.unsqueeze(1)  # [B,1]


class EASE(nn.Module):

    def __init__(self, args):
        super().__init__()

        fe = args['model']['feature_extractor']
        dim = fe['hidden_dims'][0]  # 128
        token_len = fe['token_length'][0]  # 8

        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers='bert',
            pretrained=fe['bert_pretrained']
        )

        self.proj_l = nn.Sequential(
            nn.Linear(fe['input_dims'][0], fe['hidden_dims'][0]),
            Transformer(
                num_frames=fe['input_length'][0],
                save_hidden=False,
                token_len=fe['token_length'][0],
                dim=fe['hidden_dims'][0],
                depth=fe['depth'],
                heads=fe['heads'],
                mlp_dim=fe['hidden_dims'][0]
            )
        )
        self.proj_v = nn.Sequential(
            nn.Linear(fe['input_dims'][1], fe['hidden_dims'][1]),
            Transformer(
                num_frames=fe['input_length'][1],
                save_hidden=False,
                token_len=fe['token_length'][1],
                dim=fe['hidden_dims'][1],
                depth=fe['depth'],
                heads=fe['heads'],
                mlp_dim=fe['hidden_dims'][1]
            )
        )
        self.proj_a = nn.Sequential(
            nn.Linear(fe['input_dims'][2], fe['hidden_dims'][2]),
            Transformer(
                num_frames=fe['input_length'][2],
                save_hidden=False,
                token_len=fe['token_length'][2],
                dim=fe['hidden_dims'][2],
                depth=fe['depth'],
                heads=fe['heads'],
                mlp_dim=fe['hidden_dims'][2]
            )
        )

 
        ucmi_cfg = args['model'].get('ucmi', {})
        e_depth = ucmi_cfg.get('depth', 2)
        e_heads = ucmi_cfg.get('heads', 8)
        e_mlp = ucmi_cfg.get('mlp_dim', 128)
        e_eps = ucmi_cfg.get('eps', 1e-6)


        self.ucmi_t = UCMIModule(dim=dim, depth=e_depth, heads=e_heads, mlp_dim=e_mlp, eps=e_eps, num_frames=16)
        self.ucmi_a = UCMIModule(dim=dim, depth=e_depth, heads=e_heads, mlp_dim=e_mlp, eps=e_eps, num_frames=16)
        self.ucmi_v = UCMIModule(dim=dim, depth=e_depth, heads=e_heads, mlp_dim=e_mlp, eps=e_eps, num_frames=16)


        gate_cfg = args['model'].get('gate', {})
        g_depth = gate_cfg.get('depth', 2)
        g_heads = gate_cfg.get('heads', 8)
        g_mlp = gate_cfg.get('mlp_dim', 128)

        self.gate_l = CompletenessEstimator(dim=dim, depth=g_depth, heads=g_heads, mlp_dim=g_mlp, token_len=token_len)
        self.gate_a = CompletenessEstimator(dim=dim, depth=g_depth, heads=g_heads, mlp_dim=g_mlp, token_len=token_len)
        self.gate_v = CompletenessEstimator(dim=dim, depth=g_depth, heads=g_heads, mlp_dim=g_mlp, token_len=token_len)

        fuse_cfg = args['model'].get('fusion', {})
        f_depth = fuse_cfg.get('depth', 4)
        f_heads = fuse_cfg.get('heads', 8)
        f_mlp = fuse_cfg.get('mlp_dim', 128)
        

        cmdc_cfg = args.get('loss', {}).get('cmdc', {})  
        self.cmdc_num_views = int(cmdc_cfg.get('num_views', 3))   
        self.cmdc_drop_prob = float(cmdc_cfg.get('drop_prob', 0.5))  
        self.cmdc_use_hallucinated_when_drop = bool(cmdc_cfg.get('use_hallucinated_when_drop', False))


        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.fusion = Transformer(
            num_frames=1 + 3 * token_len,  # CLS + (T,A,V tokens)
            save_hidden=False,
            token_len=None,
            dim=dim,
            depth=f_depth,
            heads=f_heads,
            mlp_dim=f_mlp
        )
        self.regressor = nn.Linear(dim, args['model']['head']['out_dim'])
    
    def _predict_from_tokens(self, h_l_tok, h_a_tok, h_v_tok):

        B = h_l_tok.size(0)
        cls = repeat(self.cls, '1 1 d -> b 1 d', b=B)
        fused_seq = torch.cat([cls, h_l_tok, h_a_tok, h_v_tok], dim=1)  # [B,1+24,D]
        fused_seq = self.fusion(fused_seq)
        feat = fused_seq[:, 0]
        pred = self.regressor(feat)
        return pred


    def _apply_subset_mask(self, h_obs, h_hall, keep_mask: torch.Tensor):

        B, T, D = h_obs.shape
        keep = keep_mask.view(B, 1, 1).to(h_obs.dtype)  # broadcast
        if self.cmdc_use_hallucinated_when_drop:
            return keep * h_obs + (1.0 - keep) * h_hall
        else:
            return keep * h_obs
    
    def sample_keep_mask(self, B, device, drop_prob):
        keep = (torch.rand(B, 3, device=device) > drop_prob).float()

        all_drop = (keep.sum(dim=1) < 0.5)  # [B]
        if all_drop.any():
            idx = all_drop.nonzero(as_tuple=True)[0]                 # [N]
            pick = torch.randint(0, 3, (idx.numel(),), device=device) # [N]

            fix = torch.zeros(B, 3, device=device)
            fix[idx, :] = 0.0
            fix[idx, pick] = 1.0


            keep = torch.where(all_drop.view(B, 1), fix, keep)

        keep_l = keep[:, 0:1]
        keep_a = keep[:, 1:2]
        keep_v = keep[:, 2:3]
        return keep_l, keep_a, keep_v, keep
    


    def forward(self, complete_input, incomplete_input, missing_rates=None):

        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        B = vision_m.size(0)

        h_l = self.proj_l(self.bertmodel(language_m))[:, :8]  # [B,8,128]
        h_a = self.proj_a(audio_m)[:, :8]                     # [B,8,128]
        h_v = self.proj_v(vision_m)[:, :8]                    # [B,8,128]


        src_for_t = torch.cat([h_a, h_v], dim=1)
        src_for_a = torch.cat([h_l, h_v], dim=1)
        src_for_v = torch.cat([h_l, h_a], dim=1)

        mu_t, sigma2_t, zhat_t, u_tok_t, u_t = self.ucmi_t(src_for_t, out_tokens=8, sample=True)
        mu_a, sigma2_a, zhat_a, u_tok_a, u_a = self.ucmi_a(src_for_a, out_tokens=8, sample=True)
        mu_v, sigma2_v, zhat_v, u_tok_v, u_v = self.ucmi_v(src_for_v, out_tokens=8, sample=True)

        if missing_rates is not None and 'l' in missing_rates:
            miss_l = missing_rates['l'].clamp(0, 1)
            miss_a = missing_rates['a'].clamp(0, 1)
            miss_v = missing_rates['v'].clamp(0, 1)
        else:
            miss_l = estimate_text_missing_rate(language_m)   # [B,1]
            miss_a = estimate_text_missing_rate(audio_m)
            miss_v = estimate_text_missing_rate(vision_m) 

        w_l = self.gate_l(h_l) * (1.0 - miss_l)     # [B,1]
        w_a = self.gate_a(h_a) * (1.0 - miss_a) 
        w_v = self.gate_v(h_v) * (1.0 - miss_v) 
        
        h_l_f = h_l * w_l.unsqueeze(-1) + zhat_t * (1.0 - w_l.unsqueeze(-1))
        h_a_f = h_a * w_a.unsqueeze(-1) + zhat_a * (1.0 - w_a.unsqueeze(-1))
        h_v_f = h_v * w_v.unsqueeze(-1) + zhat_v * (1.0 - w_v.unsqueeze(-1))


        cls = repeat(self.cls, '1 1 d -> b 1 d', b=B)
        fused_seq = torch.cat([cls, h_l_f, h_a_f, h_v_f], dim=1)   # [B, 1+24, 128]
        fused_seq = self.fusion(fused_seq)                         # [B, 1+24, 128]

        feat = fused_seq[:, 0]                                     # CLS
        pred = self.regressor(feat)                                # [B, out_dim]
        
        
        cmdc = None
        if self.training:
            pred_full = pred 
            
            preds = [pred_full]
            masks = [torch.ones(B, 3, device=pred.device)]  # (keep_l, keep_a, keep_v) = (1,1,1)

            if self.cmdc_num_views >= 2:
                keep_l, keep_a, keep_v, keep_mat = self.sample_keep_mask(B, pred.device, self.cmdc_drop_prob)

                hl = self._apply_subset_mask(h_l_f, zhat_t, keep_l)
                ha = self._apply_subset_mask(h_a_f, zhat_a, keep_a)
                hv = self._apply_subset_mask(h_v_f, zhat_v, keep_v)
                pred_s1 = self._predict_from_tokens(hl, ha, hv)

                preds.append(pred_s1)
                masks.append(keep_mat)   # [B,3]

            if self.cmdc_num_views >= 3:
                keep_l2, keep_a2, keep_v2, keep_mat2 = self.sample_keep_mask(B, pred.device, self.cmdc_drop_prob)

                hl2 = self._apply_subset_mask(h_l_f, zhat_t, keep_l2)
                ha2 = self._apply_subset_mask(h_a_f, zhat_a, keep_a2)
                hv2 = self._apply_subset_mask(h_v_f, zhat_v, keep_v2)
                pred_s2 = self._predict_from_tokens(hl2, ha2, hv2)

                preds.append(pred_s2)
                masks.append(keep_mat2)


            cmdc = {
                'preds': preds,            # list of tensors, len = num_views, each [B,out_dim]
                'masks': masks,            # list of tensors, each [B,3]
                'pred_full': pred_full,    # [B,out_dim]
            }


        complete_feats = None
        if (vision is not None) and (audio is not None) and (language is not None):
            cl = self.proj_l(self.bertmodel(language))[:, :8]
            ca = self.proj_a(audio)[:, :8]
            cv = self.proj_v(vision)[:, :8]
            # keep consistent order: (l,a,v) and also a concatenated tensor
            complete_feats = {
                'l': cl, 'a': ca, 'v': cv,
                'cat': torch.cat([cl, ca, cv], dim=1)  # [B,24,128]
            }

        return {
            'sentiment_preds': pred,
            'features': feat, 
            'cmdc': cmdc,

            'h_l': h_l, 'h_a': h_a, 'h_v': h_v,


            'h_l_f': h_l_f, 'h_a_f': h_a_f, 'h_v_f': h_v_f,
            'w_l': w_l, 'w_a': w_a, 'w_v': w_v,

            'ucmi': {
                'mu_t': mu_t, 'sigma2_t': sigma2_t, 'zhat_t': zhat_t, 'u_tok_t': u_tok_t, 'u_t': u_t,
                'mu_a': mu_a, 'sigma2_a': sigma2_a, 'zhat_a': zhat_a, 'u_tok_a': u_tok_a, 'u_a': u_a,
                'mu_v': mu_v, 'sigma2_v': sigma2_v, 'zhat_v': zhat_v, 'u_tok_v': u_tok_v, 'u_v': u_v,
            },


            'complete_feats': complete_feats
        }


def build_model(args):
    return EASE(args)
