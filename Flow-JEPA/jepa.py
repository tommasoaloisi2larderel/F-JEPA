"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        base_predictor=None,
        source_predictor=None,
        objective="flow",
        num_flow_steps=8,
        flow_source="standard_noise",
        flow_source_noise_scale=1.0,
        residual_base_loss_weight=1.0,
        learned_source_loss_weight=0.1,
        learned_source_min_scale=0.01,
        learned_source_max_scale=2.0,
        rollout_loss_weight=1.0,
        rollout_horizon=2,
        freeze_encoder=False,
        freeze_projector=False,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.base_predictor = base_predictor
        self.source_predictor = source_predictor
        self.objective = str(objective).lower()
        self.num_flow_steps = num_flow_steps
        self.flow_source = str(flow_source).lower()
        self.flow_source_noise_scale = float(flow_source_noise_scale)
        self.residual_base_loss_weight = float(residual_base_loss_weight)
        self.learned_source_loss_weight = float(learned_source_loss_weight)
        self.learned_source_min_scale = float(learned_source_min_scale)
        self.learned_source_max_scale = float(learned_source_max_scale)
        self.rollout_loss_weight = float(rollout_loss_weight)
        self.rollout_horizon = int(rollout_horizon)
        self._validate_flow_configuration()
        self.freeze_encoder = bool(freeze_encoder)
        self.freeze_projector = bool(freeze_projector)
        self.set_visual_encoder_frozen(
            self.freeze_encoder,
            self.freeze_projector,
        )
        self._flow_seed = None
        self._flow_generators = {}

    def _validate_flow_configuration(self):
        if self.objective not in {
            "flow",
            "deterministic",
            "residual_flow",
            "learned_source_flow",
            "transition_flow_rollout2",
        }:
            raise ValueError(
                "objective must be one of 'flow', 'deterministic', "
                "'residual_flow', 'learned_source_flow', or "
                "'transition_flow_rollout2'"
            )
        if self.flow_source not in {
            "standard_noise",
            "noisy_current",
        }:
            raise ValueError(
                "flow_source must be either 'standard_noise' or "
                "'noisy_current'"
            )
        if self.flow_source_noise_scale < 0:
            raise ValueError(
                "flow_source_noise_scale must be non-negative"
            )
        if self.objective == "residual_flow" and self.base_predictor is None:
            raise ValueError(
                "residual_flow requires a base_predictor module"
            )
        if (
            self.objective == "learned_source_flow"
            and self.source_predictor is None
        ):
            raise ValueError(
                "learned_source_flow requires a source_predictor module"
            )
        if self.learned_source_min_scale <= 0:
            raise ValueError("learned_source_min_scale must be positive")
        if self.learned_source_max_scale < self.learned_source_min_scale:
            raise ValueError(
                "learned_source_max_scale must be >= learned_source_min_scale"
            )
        if self.rollout_horizon < 1:
            raise ValueError("rollout_horizon must be positive")
        if (
            self.objective == "transition_flow_rollout2"
            and self.rollout_horizon != 2
        ):
            raise ValueError(
                "transition_flow_rollout2 currently expects rollout_horizon=2"
            )

    @staticmethod
    def _set_module_frozen(module, frozen):
        module.requires_grad_(not frozen)
        if frozen:
            module.eval()

    def set_visual_encoder_frozen(
        self,
        freeze_encoder=True,
        freeze_projector=True,
    ):
        """Freeze the modules that define the visual latent coordinates."""
        self.freeze_encoder = bool(freeze_encoder)
        self.freeze_projector = bool(freeze_projector)
        self._set_module_frozen(self.encoder, self.freeze_encoder)
        self._set_module_frozen(self.projector, self.freeze_projector)
        return self

    def train(self, mode=True):
        """Keep frozen visual modules and normalization statistics in eval mode."""
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
        if getattr(self, "freeze_projector", False):
            self.projector.eval()
        return self

    def set_flow_seed(self, seed):
        """Reset the dedicated RNG used for flow-matching inference."""
        self._flow_seed = None if seed is None else int(seed)
        self._flow_generators = {}
        return self

    def _get_flow_generator(self, device):
        if self._flow_seed is None:
            return None
        device = torch.device(device)
        key = str(device)
        if key not in self._flow_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._flow_seed)
            self._flow_generators[key] = generator
        return self._flow_generators[key]

    def _apply_token_module(self, module, x):
        if isinstance(module, nn.Identity):
            return x
        shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        x = module(x)
        return x.reshape(*shape[:-1], x.size(-1))

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        if hasattr(output, "last_hidden_state"):
            pixels_emb = output.last_hidden_state[:, 0]
            emb = self.projector(pixels_emb)
            emb = rearrange(emb, "(b t) d -> b t 1 d", b=b)
        else:
            emb = self._apply_token_module(self.projector, output)
            emb = rearrange(emb, "(b t) k d -> b t k d", b=b)
        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def _repeated_current_latent(self, hist_emb, pred_len):
        """Repeat the final history latent across the prediction horizon."""
        if hist_emb.ndim != 4:
            raise ValueError(
                "history embeddings must have shape (B, H, K, D)"
            )
        pred_len = int(pred_len)
        if pred_len <= 0:
            raise ValueError("prediction length must be positive")
        return hist_emb[:, -1:].expand(-1, pred_len, -1, -1)

    def _predict_with_module(
        self,
        module,
        query_emb,
        hist_emb,
        future_act_emb,
        t=None,
    ):
        if t is None:
            t = torch.zeros(
                query_emb.size(0),
                device=query_emb.device,
                dtype=query_emb.dtype,
            )
        output = module(query_emb, hist_emb, future_act_emb, t)
        return self._apply_token_module(self.pred_proj, output)

    def _deterministic_query(self, hist_emb, future_act_emb):
        return self._repeated_current_latent(hist_emb, future_act_emb.size(1))

    def _deterministic_prediction(
        self,
        hist_emb,
        future_act_emb,
        module=None,
    ):
        query = self._deterministic_query(hist_emb, future_act_emb)
        return self._predict_with_module(
            module or self.predictor,
            query,
            hist_emb,
            future_act_emb,
        )

    def _residual_base_prediction(self, hist_emb, future_act_emb):
        return self._deterministic_prediction(
            hist_emb,
            future_act_emb,
            module=self.base_predictor,
        )

    def _learned_source_distribution(self, hist_emb, future_act_emb):
        query = self._deterministic_query(hist_emb, future_act_emb)
        params = self.source_predictor(
            query,
            hist_emb,
            future_act_emb,
            torch.zeros(
                query.size(0),
                device=query.device,
                dtype=query.dtype,
            ),
        )
        mu, raw_scale = params.chunk(2, dim=-1)
        scale_span = self.learned_source_max_scale - self.learned_source_min_scale
        scale = self.learned_source_min_scale + scale_span * raw_scale.sigmoid()
        return mu, scale

    def _flow_source_from_noise(self, hist_emb, noise):
        """Transform raw Gaussian noise into the configured flow source."""
        if noise.ndim != 4:
            raise ValueError(
                "flow noise must have shape (B, P, K, D)"
            )
        if hist_emb.size(0) != noise.size(0):
            raise ValueError("history and flow-noise batch sizes must match")
        if hist_emb.shape[2:] != noise.shape[2:]:
            raise ValueError(
                "history and flow-noise latent token shapes must match"
            )
        noise_scale = float(
            getattr(self, "flow_source_noise_scale", 1.0)
        )
        source = noise * noise_scale
        flow_source = getattr(
            self,
            "flow_source",
            "standard_noise",
        )
        if flow_source == "standard_noise":
            return source
        if flow_source != "noisy_current":
            raise ValueError(f"unsupported flow source: {flow_source}")
        current_latent = self._repeated_current_latent(
            hist_emb,
            noise.size(1),
        )
        return current_latent + source

    def _transition_flow_source(self, current_emb, noise):
        if current_emb.ndim != 4 or current_emb.size(1) != 1:
            raise ValueError(
                "current transition embeddings must have shape (B, 1, K, D)"
            )
        if noise.shape != current_emb.shape:
            raise ValueError("transition noise and current embeddings must match")
        return current_emb + noise * self.flow_source_noise_scale

    def _transition_flow_loss(self, current_emb, action_emb, target_emb):
        """Flow-match one-step latent transitions z_t -> z_{t+1}."""
        if target_emb.shape != current_emb.shape:
            raise ValueError("current and target embeddings must have same shape")
        if action_emb.ndim != 3 or action_emb.size(1) != 1:
            raise ValueError(
                "transition action embeddings must have shape (B, 1, A)"
            )
        noise = torch.randn_like(target_emb)
        source = self._transition_flow_source(current_emb, noise)
        t = torch.rand(target_emb.size(0), device=target_emb.device)
        t_view = t.view(-1, 1, 1, 1)
        noisy_emb = (1 - t_view) * source + t_view * target_emb
        target_velocity = target_emb - source
        pred_velocity = self.predictor(noisy_emb, current_emb, action_emb, t)
        pred_velocity = self._apply_token_module(self.pred_proj, pred_velocity)
        return F.mse_loss(pred_velocity, target_velocity)

    def _predict_transition(self, current_emb, action_emb, noise=None):
        """Integrate one local transition flow from z_t to z_{t+1}."""
        if current_emb.ndim != 4 or current_emb.size(1) != 1:
            raise ValueError(
                "current transition embeddings must have shape (B, 1, K, D)"
            )
        if action_emb.ndim != 3 or action_emb.size(1) != 1:
            raise ValueError(
                "transition action embeddings must have shape (B, 1, A)"
            )
        if noise is None:
            generator = self._get_flow_generator(current_emb.device)
            noise = torch.randn(
                current_emb.shape,
                device=current_emb.device,
                dtype=current_emb.dtype,
                generator=generator,
            )
        x = self._transition_flow_source(current_emb, noise)
        num_steps = self.num_flow_steps
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full(
                (current_emb.size(0),),
                step / num_steps,
                device=current_emb.device,
                dtype=current_emb.dtype,
            )
            velocity = self.predictor(x, current_emb, action_emb, t)
            velocity = self._apply_token_module(self.pred_proj, velocity)
            x = x + dt * velocity
        return x

    def transition_flow_rollout2_loss(self, emb, act_emb):
        """
        Train a shared local flow z_t -> z_{t+1}, plus a two-step rollout loss.

        emb: (B, T, K, D)
        act_emb: (B, T-1 or more, A)
        """
        if emb.size(1) < 3:
            raise ValueError(
                "transition_flow_rollout2 requires at least three latent frames"
            )
        max_transitions = min(emb.size(1) - 1, act_emb.size(1))
        if max_transitions < 2:
            raise ValueError(
                "transition_flow_rollout2 requires at least two actions"
            )

        current = emb[:, :max_transitions]
        target = emb[:, 1 : max_transitions + 1]
        actions = act_emb[:, :max_transitions]
        b, n, k, d = current.shape
        local_loss = self._transition_flow_loss(
            current.reshape(b * n, 1, k, d),
            actions.reshape(b * n, 1, actions.size(-1)),
            target.reshape(b * n, 1, k, d),
        )

        z1_hat = self._predict_transition(
            emb[:, :1],
            act_emb[:, :1],
        )
        z2_hat = self._predict_transition(
            z1_hat,
            act_emb[:, 1:2],
        )
        rollout_loss = F.mse_loss(z2_hat, emb[:, 2:3])
        total = local_loss + self.rollout_loss_weight * rollout_loss
        return {
            "pred_loss": total,
            "flow_local_loss": local_loss,
            "rollout_loss": rollout_loss,
        }

    def flow_loss(self, hist_emb, future_act_emb, target_emb):
        """
        Compute flow-matching loss for future latent prediction.
        hist_emb: (B, H, K, D)
        future_act_emb: (B, F, A_emb)
        target_emb: (B, P, K, D)
        """
        if self.objective == "transition_flow_rollout2":
            if hist_emb.size(1) != 1:
                raise ValueError(
                    "transition_flow_rollout2 one-step loss expects "
                    "history_size=1"
                )
            return self._transition_flow_loss(
                hist_emb,
                future_act_emb[:, :1],
                target_emb[:, :1],
            )

        if self.objective == "deterministic":
            pred_emb = self._deterministic_prediction(hist_emb, future_act_emb)
            return F.mse_loss(pred_emb, target_emb)

        noise = torch.randn_like(target_emb)
        aux_loss = target_emb.new_tensor(0.0)
        if self.objective == "residual_flow":
            base_pred = self._residual_base_prediction(hist_emb, future_act_emb)
            aux_loss = F.mse_loss(base_pred, target_emb)
            flow_source = base_pred.detach() + noise * self.flow_source_noise_scale
        elif self.objective == "learned_source_flow":
            source_mu, source_scale = self._learned_source_distribution(
                hist_emb,
                future_act_emb,
            )
            aux_loss = F.mse_loss(source_mu, target_emb)
            flow_source = source_mu.detach() + source_scale.detach() * noise
        else:
            flow_source = self._flow_source_from_noise(hist_emb, noise)
        t = torch.rand(target_emb.size(0), device=target_emb.device)
        t_view = t.view(-1, 1, 1, 1)
        noisy_emb = (1 - t_view) * flow_source + t_view * target_emb
        target_velocity = target_emb - flow_source
        pred_velocity = self.predictor(noisy_emb, hist_emb, future_act_emb, t)
        pred_velocity = self._apply_token_module(self.pred_proj, pred_velocity)
        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        if self.objective == "residual_flow":
            return flow_loss + self.residual_base_loss_weight * aux_loss
        if self.objective == "learned_source_flow":
            return flow_loss + self.learned_source_loss_weight * aux_loss
        return flow_loss

    def predict(self, hist_emb, future_act_emb, horizon=None, num_steps=None, noise=None):
        """Generate a future latent trajectory by Euler integration."""
        future_len = horizon or future_act_emb.size(1)
        if future_act_emb.size(1) < future_len:
            raise ValueError("future_act_emb is shorter than requested horizon")
        if future_act_emb.size(1) > future_len:
            future_act_emb = future_act_emb[:, :future_len]
        if self.objective == "deterministic":
            return self._deterministic_prediction(hist_emb, future_act_emb)
        if self.objective == "transition_flow_rollout2":
            current = hist_emb[:, -1:]
            preds = []
            for step in range(future_len):
                current = self._predict_transition(
                    current,
                    future_act_emb[:, step : step + 1],
                )
                preds.append(current)
            return torch.cat(preds, dim=1)
        num_steps = num_steps or self.num_flow_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        b, _, query_num, dim = hist_emb.shape
        if noise is None:
            generator = self._get_flow_generator(hist_emb.device)
            noise = torch.randn(
                b,
                future_len,
                query_num,
                dim,
                device=hist_emb.device,
                dtype=hist_emb.dtype,
                generator=generator,
            )
        if self.objective == "residual_flow":
            base_pred = self._residual_base_prediction(hist_emb, future_act_emb)
            x = base_pred + noise * self.flow_source_noise_scale
        elif self.objective == "learned_source_flow":
            source_mu, source_scale = self._learned_source_distribution(
                hist_emb,
                future_act_emb,
            )
            x = source_mu + source_scale * noise
        else:
            x = self._flow_source_from_noise(hist_emb, noise)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full(
                (b,),
                step / num_steps,
                device=hist_emb.device,
                dtype=hist_emb.dtype,
            )
            velocity = self.predictor(x, hist_emb, future_act_emb, t)
            velocity = self._apply_token_module(self.pred_proj, velocity)
            x = x + dt * velocity
        return x

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 1):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        B, S, horizon = action_sequence.shape[:3]

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init.pop("action", None)
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(action_sequence, "b s ... -> (b s) ...")
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -history_size:]
        pred_emb = self.predict(emb_trunc, act_emb, horizon=horizon)
        pred_rollout = rearrange(
            pred_emb, "(b s) ... -> b s ...", b=B, s=S
        )
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        pred_final = pred_emb[:, :, -1]
        goal_final = goal_emb[:, -1].unsqueeze(1).expand_as(pred_final)
        cost = F.mse_loss(
            pred_final,
            goal_final.detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_final.ndim)))

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
