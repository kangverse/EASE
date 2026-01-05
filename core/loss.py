
from torch import nn
import torch
import torch.nn.functional as F


class EASELoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.sigma_sp = float(args['base'].get('sigma', 1.0))
        self.w_ucmi = float(args['base'].get('lambda_ucmi', 1.0))
        self.w_uada = float(args['base'].get('lambda_uada', 1.0))
        self.w_anchor = float(args['base'].get('lambda_anchor', 1.0))

        uada = args.get('loss', {}).get('uada', {})
        self.uada_sigma = float(uada.get('sigma', 1.0))
        self.uada_lambda = float(uada.get('lambda', 1.0))
        self.eps = 1e-8

        anchor = args.get('loss', {}).get('anchor', {})
        self.anchor_sigma0 = float(anchor.get('sigma0', 0.1))
        self.anchor_beta = float(anchor.get('beta', 1.0))
        self.anchor_ema = float(anchor.get('ema', 0.99))
        
        cmdc = args.get('loss', {}).get('cmdc', {})
        self.w_cons  = float(cmdc.get('lambda_cons', 1.0))
        self.w_asym  = float(cmdc.get('lambda_asym', 1.0))
        self.cons_temperature = float(cmdc.get('temperature', 1.0))  # classification


        edges = torch.tensor(anchor.get('bin_edges', [-0.1, 0.1]), dtype=torch.float32)
        self.register_buffer("bin_edges", edges)

        self.K = len(edges) + 1
        D = int(args['model']['feature_extractor']['hidden_dims'][0]) 


        self.register_buffer('prior_mu_t', torch.zeros(self.K, D))
        self.register_buffer('prior_var_t', torch.ones(self.K, D))
        self.register_buffer('prior_count_t', torch.zeros(self.K))

        self.register_buffer('prior_mu_a', torch.zeros(self.K, D))
        self.register_buffer('prior_var_a', torch.ones(self.K, D))
        self.register_buffer('prior_count_a', torch.zeros(self.K))

        self.register_buffer('prior_mu_v', torch.zeros(self.K, D))
        self.register_buffer('prior_var_v', torch.ones(self.K, D))
        self.register_buffer('prior_count_v', torch.zeros(self.K))

        self.MSE_Fn = nn.MSELoss()


        self.anchor_var_min = float(anchor.get('var_min', 1e-3)) 
    
    
    def _kl_div(self, p_logit, q_logit, T=1.0):
        # KL( softmax(p/T) || softmax(q/T) )
        p = F.log_softmax(p_logit / T, dim=-1)
        q = F.softmax(q_logit / T, dim=-1)
        return F.kl_div(p, q, reduction='batchmean')


    def _consistency_loss(self, preds, pred_full):

        if preds is None or len(preds) <= 1:
            return torch.tensor(0.0, device=pred_full.device), torch.tensor(0.0, device=pred_full.device)

        out_dim = pred_full.size(-1)

        if out_dim == 1:

            l_cons = 0.0
            cnt = 0
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    l_cons = l_cons + F.mse_loss(preds[i], preds[j])
                    cnt += 1
            l_cons = l_cons / max(cnt, 1)


            l_asym = 0.0
            for k in range(1, len(preds)):
                l_asym = l_asym + F.mse_loss(preds[k], pred_full.detach())
            l_asym = l_asym / max(len(preds) - 1, 1)
            return l_cons, l_asym


        T = self.cons_temperature
        l_cons = 0.0
        cnt = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                l_cons = l_cons + self._kl_div(preds[i], preds[j], T=T)
                cnt += 1
        l_cons = l_cons / max(cnt, 1)

        l_asym = 0.0
        for k in range(1, len(preds)):
            l_asym = l_asym + self._kl_div(preds[k], pred_full.detach(), T=T)
        l_asym = l_asym / max(len(preds) - 1, 1)
        return l_cons, l_asym


    def _ensure_device(self, device):

        if self.bin_edges.device != device:
            self.bin_edges = self.bin_edges.to(device)

        for name in [
            'prior_mu_t', 'prior_var_t', 'prior_count_t',
            'prior_mu_a', 'prior_var_a', 'prior_count_a',
            'prior_mu_v', 'prior_var_v', 'prior_count_v',
        ]:
            buf = getattr(self, name)
            if buf.device != device:
                setattr(self, name, buf.to(device))

    def _label_to_bin(self, y):
        y = y.view(-1)
        return torch.bucketize(y, self.bin_edges)

    
    def _gaussian_nll(self, mu, sigma2, z):
        nll = 0.5 * (
            torch.log(2 * torch.pi * sigma2 + self.eps)
            + (z - mu) ** 2 / (sigma2 + self.eps)
        )
        return nll.mean()

    def _uada_kernel(self, z, zhat, u, uhat):
        def pdist2(x, y):
            x2 = (x ** 2).sum(dim=1, keepdim=True)
            y2 = (y ** 2).sum(dim=1, keepdim=True).t()
            xy = x @ y.t()
            return (x2 + y2 - 2 * xy).clamp_min(0.0)

        mean_u_zh = 0.5 * (u + uhat.t())
        psi_zh = 1.0 + self.uada_lambda * mean_u_zh

        mean_u_zz = 0.5 * (u + u.t())
        psi_zz = 1.0 + self.uada_lambda * mean_u_zz

        mean_u_hh = 0.5 * (uhat + uhat.t())
        psi_hh = 1.0 + self.uada_lambda * mean_u_hh

        dist_zz = pdist2(z, z)
        dist_zh = pdist2(z, zhat)
        dist_hh = pdist2(zhat, zhat)

        denom_zz = 2.0 * (self.uada_sigma ** 2) * psi_zz
        denom_zh = 2.0 * (self.uada_sigma ** 2) * psi_zh
        denom_hh = 2.0 * (self.uada_sigma ** 2) * psi_hh

        Kzz = torch.exp(-dist_zz / (denom_zz + self.eps))
        Kzh = torch.exp(-dist_zh / (denom_zh + self.eps))
        Khh = torch.exp(-dist_hh / (denom_hh + self.eps))
        return Kzz, Kzh, Khh

    def _cs_divergence(self, Kzz, Kzh, Khh):
        num = Kzh.sum()
        den = torch.sqrt(Kzz.sum() * Khh.sum() + self.eps)
        return -torch.log(num / (den + self.eps) + self.eps)

    @torch.no_grad()
    def _update_prior_modality(self, z_complete, y_bin, modality: str):

        if modality == 't':
            prior_mu, prior_var, prior_count = self.prior_mu_t, self.prior_var_t, self.prior_count_t
        elif modality == 'a':
            prior_mu, prior_var, prior_count = self.prior_mu_a, self.prior_var_a, self.prior_count_a
        elif modality == 'v':
            prior_mu, prior_var, prior_count = self.prior_mu_v, self.prior_var_v, self.prior_count_v
        else:
            raise ValueError(f"Unknown modality: {modality}")

        y_bin = y_bin.to(prior_mu.device)
        for k in range(self.K):
            mask = (y_bin == k)
            if mask.any():
                zk = z_complete[mask]
                mu = zk.mean(dim=0)
                var = zk.var(dim=0, unbiased=False).clamp_min(1e-4)

                if prior_count[k] < 1:
                    prior_mu[k].copy_(mu)
                    prior_var[k].copy_(var)
                else:
                    prior_mu[k].mul_(self.anchor_ema).add_(mu, alpha=(1 - self.anchor_ema))
                    prior_var[k].mul_(self.anchor_ema).add_(var, alpha=(1 - self.anchor_ema))

                prior_count[k] += mask.sum().float()

    def _anchor_loss_modality(self, zhat_mu, u, y_bin, modality: str):

        device = zhat_mu.device
        y_bin = y_bin.to(device)

        if modality == 't':
            mu_y = self.prior_mu_t[y_bin]
            var_y = self.prior_var_t[y_bin]
        elif modality == 'a':
            mu_y = self.prior_mu_a[y_bin]
            var_y = self.prior_var_a[y_bin]
        elif modality == 'v':
            mu_y = self.prior_mu_v[y_bin]
            var_y = self.prior_var_v[y_bin]
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if u.dim() == 1:
            u = u.view(-1, 1)

        sigma_hat = (self.anchor_sigma0 ** 2) + self.anchor_beta * u
        sigma_hat = sigma_hat.clamp_min(self.anchor_var_min)
        var_hat = sigma_hat.expand_as(zhat_mu)

        var_sum = (var_hat + var_y).clamp_min(1e-6)

        cen = ((zhat_mu - mu_y) ** 2 / var_sum).sum(dim=1)

        shape = (
            torch.log(var_sum).sum(dim=1)
            - 0.5 * torch.log(2.0 * var_hat.clamp_min(1e-6)).sum(dim=1)
            - 0.5 * torch.log(2.0 * var_y.clamp_min(1e-6)).sum(dim=1)
        )
        return (cen + shape).mean()

    def forward(self, out, label):
        device = out['sentiment_preds'].device
        self._ensure_device(device)

        label_y = label['sentiment_labels'].to(device)
        l_sp = self.MSE_Fn(out['sentiment_preds'], label_y)


        if out.get('complete_feats', None) is None:
            l_cons = torch.tensor(0.0, device=device)
            l_asym = torch.tensor(0.0, device=device)
            if out.get('cmdc', None) is not None:
                preds = out['cmdc'].get('preds', None)
                pred_full = out['cmdc'].get('pred_full', out['sentiment_preds'])
                l_cons, l_asym = self._consistency_loss(preds, pred_full)

            loss = self.sigma_sp * l_sp + self.w_cons * l_cons + self.w_asym * l_asym
            return {'loss': loss, 'l_sp': l_sp, 'l_ucmi': 0.0, 'l_uada': 0.0, 'l_anchor': 0.0,
                    'l_cons': l_cons, 'l_asym': l_asym}


        ec = out['ucmi']
        cf = out['complete_feats']

        # complete feats
        z_l_tok = cf['l'].to(device)  # [B, T, D]
        z_a_tok = cf['a'].to(device)
        z_v_tok = cf['v'].to(device)

        z_l = z_l_tok.mean(dim=1)     # [B, D]
        z_a = z_a_tok.mean(dim=1)
        z_v = z_v_tok.mean(dim=1)

        l_ucmi = (
            self._gaussian_nll(ec['mu_t'], ec['sigma2_t'], z_l_tok) +
            self._gaussian_nll(ec['mu_a'], ec['sigma2_a'], z_a_tok) +
            self._gaussian_nll(ec['mu_v'], ec['sigma2_v'], z_v_tok)
        ) / 3.0

        zhat_l = ec['zhat_t'].mean(dim=1)
        zhat_a = ec['zhat_a'].mean(dim=1)
        zhat_v = ec['zhat_v'].mean(dim=1)

        u_l = ec['u_t'].to(device)
        u_a = ec['u_a'].to(device)
        u_v = ec['u_v'].to(device)

        u0 = torch.zeros_like(u_l)

        l_uada = 0.0
        for (z, zh, u) in [(z_l, zhat_l, u_l), (z_a, zhat_a, u_a), (z_v, zhat_v, u_v)]:
            Kzz, Kzh, Khh = self._uada_kernel(z, zh, u0, u)
            l_uada = l_uada + self._cs_divergence(Kzz, Kzh, Khh)
        l_uada = l_uada / 3.0

        y_bin = self._label_to_bin(label_y)  # [B]

        self._update_prior_modality(z_l.detach(), y_bin, modality='t')
        self._update_prior_modality(z_a.detach(), y_bin, modality='a')
        self._update_prior_modality(z_v.detach(), y_bin, modality='v')

        zhat_mu_l = ec['mu_t'].mean(dim=1)
        zhat_mu_a = ec['mu_a'].mean(dim=1)
        zhat_mu_v = ec['mu_v'].mean(dim=1)

        l_anchor = (
            self._anchor_loss_modality(zhat_mu_l, u_l, y_bin, modality='t') +
            self._anchor_loss_modality(zhat_mu_a, u_a, y_bin, modality='a') +
            self._anchor_loss_modality(zhat_mu_v, u_v, y_bin, modality='v')
        ) / 3.0
        

        l_cons = torch.tensor(0.0, device=device)
        l_asym = torch.tensor(0.0, device=device)
        if out.get('cmdc', None) is not None:
            preds = out['cmdc'].get('preds', None)
            pred_full = out['cmdc'].get('pred_full', out['sentiment_preds'])
            l_cons, l_asym = self._consistency_loss(preds, pred_full)


        loss = self.sigma_sp * l_sp + self.w_ucmi * l_ucmi + self.w_uada * l_uada + self.w_anchor * l_anchor + self.w_cons * l_cons + self.w_asym * l_asym

        return {'loss': loss, 'l_sp': l_sp, 'l_ucmi': l_ucmi, 'l_uada': l_uada, 'l_anchor': l_anchor, 'l_cons': l_cons, 'l_asym': l_asym}
