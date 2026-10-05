"""Variantes de Flow-JEPA pour séparer ses deux ingrédients (expériences Push-T et TwoRoom).

Leur Flow-JEPA (variante A) combine deux choses :
- la prédiction des 5 positions futures d'un coup (« joint ») ;
- le flow matching : partir d'un brouillon aléatoire et le rendre net en 8 pas (« flow »).
VariantJEPA garde leur réseau et leur classe JEPA (jepa.py) et permet de retirer l'un ou l'autre :

    joint=True,  flow=True   A : leur Flow-JEPA (on utilise alors directement leur classe JEPA)
    joint=True,  flow=False  B : 5 positions d'un coup, calculées directement
    joint=False, flow=True   C : pas à pas, avec flow matching
    joint=False, flow=False  D : pas à pas, calculé directement (comme LeWM, mais avec leur réseau)

Calcul direct (flow=False), sans brouillon aléatoire : on part de la moyenne de leur point de départ
(z_t répété avec le départ « noisy_current », 0 avec le départ standard), au temps tau = 0,
et le réseau prédit l'écart en une seule passe.
C'est A sans le tirage du brouillon, sans le tirage de tau et en un seul pas.

Pas à pas (joint=False) : à l'entraînement, chaque transition z_k -> z_k+1 de la fenêtre devient un
exemple, à partir du vrai z_k (comme LeWM). À la planification, chaque position est prédite à partir
de la prédiction précédente.
"""

import torch
import torch.nn.functional as F

from jepa import JEPA  # leur classe (third_party/Flow-JEPA/jepa.py)


class VariantJEPA(JEPA):
    def __init__(self, *args, joint=True, flow=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint, self.flow = bool(joint), bool(flow)

    def _direct(self, hist_emb, future_act_emb, horizon):
        """Prédiction directe, en une passe : départ = moyenne de leur source, tau = 0."""
        b, _, k, d = hist_emb.shape
        zeros = torch.zeros(b, horizon, k, d, device=hist_emb.device, dtype=hist_emb.dtype)
        start = self._flow_source_from_noise(hist_emb, zeros)  # leur fonction, avec un bruit nul
        tau = torch.zeros(b, device=hist_emb.device, dtype=hist_emb.dtype)
        step = self.predictor(start, hist_emb, future_act_emb, tau)
        return start + self._apply_token_module(self.pred_proj, step)

    def flow_loss(self, hist_emb, future_act_emb, target_emb):
        """Perte d'entraînement. Le nom est celui de leur méthode, pour que leur code d'entraînement marche tel quel.
        hist_emb : (B, 1, K, D)   future_act_emb : (B, 5, A)   target_emb : (B, 5, K, D)"""
        if not self.joint:
            # Pas à pas : les 5 transitions de la fenêtre deviennent 5 exemples, chacun partant du vrai z_k.
            context = torch.cat([hist_emb[:, -1:], target_emb[:, :-1]], dim=1)  # z_t, ..., z_t+4
            b, p = context.shape[:2]
            hist_emb = context.reshape(b * p, 1, *context.shape[2:])
            future_act_emb = future_act_emb.reshape(b * p, 1, -1)
            target_emb = target_emb.reshape(b * p, 1, *target_emb.shape[2:])
        if self.flow:
            return super().flow_loss(hist_emb, future_act_emb, target_emb)
        return F.mse_loss(self._direct(hist_emb, future_act_emb, target_emb.size(1)), target_emb)

    def predict(self, hist_emb, future_act_emb, horizon=None, num_steps=None, noise=None):
        """Trajectoire prédite pour la planification (appelée par leur rollout)."""
        horizon = horizon or future_act_emb.size(1)
        if self.joint:
            if self.flow:
                return super().predict(hist_emb, future_act_emb, horizon, num_steps, noise)
            return self._direct(hist_emb, future_act_emb[:, :horizon], horizon)
        x, outs = hist_emb[:, -1:], []
        for k in range(horizon):
            action = future_act_emb[:, k : k + 1]
            if self.flow:
                x = super().predict(x, action, horizon=1, num_steps=num_steps)  # 8 pas d'Euler
            else:
                x = self._direct(x, action, 1)
            outs.append(x)
        return torch.cat(outs, dim=1)
