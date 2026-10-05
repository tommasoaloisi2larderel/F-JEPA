"""Entraînement rapide d'un prédicteur sur un encodeur LeWM gelé (TwoRoom ou Push-T).

L'encodeur gelé donne toujours le même vecteur pour la même image. Donc :
1. on passe chaque image du dataset UNE seule fois dans l'encodeur (quelques minutes) ;
2. on entraîne le prédicteur sur ces vecteurs, entièrement sur le GPU.

Modèles (--model), tous avec le même encodeur LeWM gelé et les mêmes vecteurs :
  joint_flow  A : leur Flow-JEPA, leur classe JEPA telle quelle
  joint_det   B : leur réseau, 5 positions d'un coup, sans bruit      (voir variants.py)
  ar_flow     C : leur réseau, pas à pas, avec flow matching
  ar_det      D : leur réseau, pas à pas, sans bruit
  lewm          : LeWM réentraîné avec sa propre recette (son réseau, 3 images d'historique, dropout 0,1)
Les réglages de chaque environnement (--env) sont ceux de l'article : voir ENVS.

Le modèle, la perte, les réglages et le format de sauvegarde sont ceux des repos officiels
(Flow-JEPA : config/train/fjepa.yaml, jepa.py, train.py ; LeWM : config.json de son checkpoint et sa perte).
Seul le chargement des données change : leur chargeur relit et décompresse 30 images par exemple,
à chaque epoch, sur le processeur.

Usage :
  python scripts/train_frozen_fast.py [--env tworoom] [--model joint_flow] [--epochs 20] [--seed 0]
  python scripts/train_frozen_fast.py --env pusht --encode-only     # calcule seulement les vecteurs
"""

import argparse
import contextlib
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

# Mêmes chemins par défaut que scripts/env.sh. À définir avant d'importer stable_worldmodel.
os.environ.setdefault("STABLEWM_HOME", "/workspace/stable-wm")

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402, F401  (décompression Blosc utilisée par le fichier du dataset)
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
FJEPA_DIR = REPO_DIR / "third_party" / "Flow-JEPA"
sys.path.insert(0, str(FJEPA_DIR))

import stable_pretraining as spt  # noqa: E402  (enregistre aussi le résolveur ${eval:...} de leurs configs)
import stable_worldmodel as swm  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402

from train import initialize_visual_encoder_from_checkpoint  # noqa: E402  (leur train.py)

STABLEWM_HOME = Path(os.environ["STABLEWM_HOME"])
ENCODE_BLOCK = 500  # images encodées à la fois : 5 blocs de compression de 100 images
READ_WORKERS = 4  # processus qui lisent et décompressent le fichier en parallèle

# Réglages de l'article pour chaque environnement (texte de l'article, annexe A).
ENVS = {
    "tworoom": {
        "data": "tworoom",
        "lewm": "lewm-tworooms",
        "overrides": ["loss.sigreg.weight=0.1"],  # le reste = config par défaut de leur repo
    },
    "pusht": {
        "data": "pusht",
        "lewm": "lewm-pusht",
        "overrides": [
            "loss.sigreg.weight=0.09",
            "flow_matching.source=noisy_current",  # départ du flow : N(z_t, 0,5²)
            "flow_matching.source_noise_scale=0.5",
            "model.predictor.causal_self_attention=true",
            "model.predictor.dropout=0.0",
        ],
    },
}
# (prédiction d'un coup, flow matching) pour chaque variante de leur réseau
MODELS = {"joint_flow": (True, True), "joint_det": (True, False), "ar_flow": (False, True), "ar_det": (False, False)}

LOG_FILE = REPO_DIR / "logs" / "train_frozen.log"  # remplacé par un fichier par entraînement dans main()


def lewm_dir(env):
    return STABLEWM_HOME / "checkpoints" / ENVS[env]["lewm"]


def emb_cache(env):
    return STABLEWM_HOME / "cache" / f"{env}_lewm_emb.pt"


LEWM_DIR, EMB_CACHE = lewm_dir("tworoom"), emb_cache("tworoom")  # valeurs du test TwoRoom (smoke_test.py)


def log(msg):
    print(msg, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def fmt(n):
    """920809 -> '920 809'."""
    return f"{n:,}".replace(",", " ")


def autocast(device):
    """bf16 sur GPU, comme `precision: bf16` dans leur config. Rien sur CPU."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---------- Config, données, modèle : comme leurs train.py ----------


def load_cfg(epochs, env="tworoom", model="joint_flow"):
    """Leur config d'entraînement, avec les réglages de l'article pour cet environnement, l'encodeur LeWM gelé
    (option prévue par leur README) et le modèle demandé."""
    overrides = [
        f"data={ENVS[env]['data']}",
        f"trainer.max_epochs={epochs}",
        "pretrained_visual_encoder.enabled=true",
        f"pretrained_visual_encoder.checkpoint={lewm_dir(env)}",
        "pretrained_visual_encoder.freeze=true",
        *ENVS[env]["overrides"],
    ]
    if model == "lewm":
        overrides += ["history_size=3", "num_preds=1"]  # la recette de LeWM : 3 images d'historique, 1 pas prédit
    with initialize_config_dir(config_dir=str(FJEPA_DIR / "config" / "train"), version_base=None):
        cfg = compose(config_name="fjepa", overrides=overrides)
    with open_dict(cfg):
        cfg.env, cfg.model_name = env, model
        if model == "lewm":  # le réseau de LeWM tel que dans son checkpoint (prédicteur, dropout 0,1...)
            cfg.model = OmegaConf.create(json.loads((lewm_dir(env) / "config.json").read_text()))
        elif model != "joint_flow":
            cfg.model._target_ = "variants.VariantJEPA"
            cfg.model.joint, cfg.model.flow = MODELS[model]
    return cfg


def load_dataset(cfg):
    """Leur dataset : des fenêtres de num_steps images espacées de frameskip pas (6 et 5 pour Flow-JEPA)."""
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop("name")
    return swm.data.load_dataset(name, transform=None, cache_dir=str(STABLEWM_HOME), **dataset_cfg)


def is_lewm(cfg):
    return cfg.get("model_name", "joint_flow") == "lewm"


def build_model(cfg, dataset, device):
    """Le modèle demandé, avec l'encodeur et le projecteur LeWM chargés puis gelés."""
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = dataset.frameskip * dataset.get_dim("action")
        if not is_lewm(cfg):
            cfg.model.freeze_encoder = True
            cfg.model.freeze_projector = True
    model = hydra.utils.instantiate(cfg.model)
    initialize_visual_encoder_from_checkpoint(model, lewm_dir(cfg.get("env", "tworoom")))
    if is_lewm(cfg):  # LeWM n'a pas de fonction de gel : on le fait à la main
        for module in (model.encoder, model.projector):
            module.requires_grad_(False)
    else:
        model.set_visual_encoder_frozen(True, True)
    return set_train_mode(model.to(device), cfg, True)


def set_train_mode(model, cfg, training):
    """Mode entraînement (dropout actif) ou évaluation. L'encodeur et le projecteur gelés restent en évaluation."""
    model.train(training)
    if is_lewm(cfg):
        model.encoder.eval()
        model.projector.eval()
    return model


# ---------- Étape 1 : passer chaque image une seule fois dans l'encodeur ----------


class PixelBlocks(torch.utils.data.Dataset):
    """Lit les images dans l'ordre, par lots alignés sur les blocs de compression du fichier.
    Chaque bloc n'est décompressé qu'une seule fois."""

    def __init__(self, h5_path, n_frames):
        self.h5_path, self.n_frames, self.file = h5_path, n_frames, None

    def __len__(self):
        return math.ceil(self.n_frames / ENCODE_BLOCK)

    def __getitem__(self, i):
        if self.file is None:  # un fichier ouvert par processus de lecture
            self.file = h5py.File(self.h5_path, "r")
        return torch.from_numpy(self.file["pixels"][i * ENCODE_BLOCK : (i + 1) * ENCODE_BLOCK])


@torch.no_grad()
def encode_pixels(model, pixels, device):
    """Images uint8 (N, 224, 224, 3) -> vecteurs (N, 192).
    Même prétraitement que leur get_img_preprocessor : [0, 255] -> [0, 1] -> normalisation ImageNet.
    Leur Resize(224) ne change rien : les images font déjà 224 x 224."""
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, 3, 1, 1)
    x = pixels.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div(255)
    x = (x - mean) / std
    with autocast(device):
        emb = model.encode({"pixels": x.unsqueeze(1)})["emb"]  # leur fonction : (N, 1, 1, 192)
    return emb.flatten(1)


def encode_all_frames(model, h5_path, n_frames, device):
    loader = torch.utils.data.DataLoader(
        PixelBlocks(h5_path, n_frames),
        batch_size=None,
        num_workers=READ_WORKERS,
        pin_memory=device == "cuda",
    )
    embs, done, start, next_report = None, 0, time.time(), 0.1
    for pixels in loader:
        emb = encode_pixels(model, pixels, device)
        if embs is None:  # bf16 sur GPU : déjà la précision des vecteurs pendant leur entraînement
            embs = torch.empty(n_frames, emb.shape[-1], dtype=emb.dtype, device=device)
        embs[done : done + len(emb)] = emb
        done += len(emb)
        if done / n_frames >= next_report:
            speed = done / (time.time() - start)
            log(f"  encodage {done / n_frames:4.0%} | {fmt(int(speed))} images/s | reste ~{(n_frames - done) / speed / 60:.1f} min")
            next_report += 0.1
    return embs


def load_or_encode(model, dataset, device, cache=EMB_CACHE):
    """Vecteurs de toutes les images. Gardés sur le disque, et partagés par tous les modèles d'un même
    environnement : ils utilisent tous le même encodeur LeWM gelé."""
    with h5py.File(dataset.h5_path, "r") as f:
        n_frames = f["pixels"].shape[0]
    if cache.exists():
        embs = torch.load(cache)
        if embs.shape[0] == n_frames:
            log(f"Étape 1 : vecteurs déjà calculés, relus depuis {cache}")
            return embs.to(device)
    log(f"Étape 1 : encodage des {fmt(n_frames)} images, une seule fois")
    embs = encode_all_frames(model, dataset.h5_path, n_frames, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    torch.save(embs.cpu(), tmp)
    os.replace(tmp, cache)  # écriture atomique : un fichier à moitié écrit ne passe jamais pour complet
    return embs


# ---------- Étape 2 : entraîner le prédicteur sur ces vecteurs ----------


def normalized_actions(dataset, device):
    """Actions centrées-réduites comme leur get_column_normalizer (utils.py).
    Les NaN (fins d'épisode) deviennent 0, comme torch.nan_to_num dans leur flow_jepa_forward."""
    raw = torch.from_numpy(np.array(dataset.get_col_data("action")))
    valid = raw[~torch.isnan(raw).any(dim=1)]
    mean, std = valid.mean(0, keepdim=True), valid.std(0, keepdim=True)
    return torch.nan_to_num(((raw - mean) / std).float(), 0.0).to(device)


def make_batch_fn(dataset, embs, actions, device, token_dim=True):
    """Renvoie une fonction : numéros d'exemples -> (vecteurs, actions), comme un batch de leur DataLoader.
    token_dim : vecteurs en (B, T, 1, D) pour Flow-JEPA, en (B, T, D) pour LeWM."""
    fs, steps = dataset.frameskip, dataset.num_steps  # 5 et 6 pour Flow-JEPA (5 et 4 pour LeWM)
    # Position dans le fichier du début de chaque exemple (même liste que leur dataset).
    starts = torch.tensor([dataset.offsets[ep] + s for ep, s in dataset.clip_indices], device=device)
    frame_offsets = torch.arange(0, steps * fs, fs, device=device)  # 0, 5, 10... : une image sur 5
    action_offsets = torch.arange(steps * fs, device=device)  # 0, 1, 2... : toutes les actions

    def get_batch(ids):
        g = starts[ids]
        emb = embs[g[:, None] + frame_offsets]  # (B, T, 192)
        act = actions[g[:, None] + action_offsets].reshape(len(ids), steps, -1)  # (B, T, 10) : 5 actions par pas
        return (emb.unsqueeze(2) if token_dim else emb), act

    return get_batch


def make_loss_fn(model, cfg):
    """La perte d'entraînement, comme dans leurs train.py. SIGReg est omis : l'encodeur est gelé,
    ce terme ne change aucun gradient. Les réglages sont lus une fois (plus simple pour torch.compile)."""
    h, p, lewm = int(cfg.history_size), int(cfg.num_preds), is_lewm(cfg)

    def loss_fn(emb, act):
        act_emb = model.action_encoder(act)
        if lewm:  # lejepa_forward de LeWM : prédire l'image suivante à chacune des 3 positions
            pred = model.predict(emb[:, :h], act_emb[:, :h])
            return (pred - emb[:, p:]).pow(2).mean()
        return model.flow_loss(emb[:, :h], act_emb[:, h - 1 : h - 1 + p], emb[:, h : h + p])

    return loss_fn


def make_optimizer(model, cfg, total_steps, device):
    """AdamW, puis warmup linéaire (1 % des pas) et décroissance cosinus : les réglages par défaut
    de stable_pretraining. Le taux d'apprentissage change à chaque pas, comme dans leur boucle."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, fused=device == "cuda"
    )
    sched = LinearWarmupCosineAnnealingLR(opt, warmup_steps=max(1, int(0.01 * total_steps)), max_steps=total_steps)
    return params, opt, sched


def train_step(model, cfg, opt, params, emb, act, device, sched=None, loss_fn=None):
    loss_fn = loss_fn or make_loss_fn(model, cfg)
    with autocast(device):
        loss = loss_fn(emb, act)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, cfg.trainer.gradient_clip_val)
    opt.step()
    if sched is not None:
        sched.step()
    return loss.detach()


@torch.no_grad()
def validation_loss(model, cfg, get_batch, ids, device, batch_size=1024):
    set_train_mode(model, cfg, False)
    loss_fn = make_loss_fn(model, cfg)
    total = torch.zeros((), device=device)
    for s in range(0, len(ids), batch_size):
        emb, act = get_batch(ids[s : s + batch_size])
        with autocast(device):
            total += loss_fn(emb, act).float() * len(emb)
    set_train_mode(model, cfg, True)
    return (total / len(ids)).item()


def choose_compile(mode, model, cfg, loss_fn, get_batch, train_ids, device):
    """Renvoie la fonction de perte à utiliser : compilée par torch.compile si ça fait gagner du temps.
    En mode auto, on mesure 30 pas avec et sans, puis on remet le modèle dans son état de départ."""
    if device != "cuda" or mode == "off":
        return loss_fn
    compiled = torch.compile(loss_fn)
    if mode == "on":
        return compiled
    state = copy.deepcopy(model.state_dict())
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    emb, act = get_batch(train_ids[: cfg.loader.batch_size])
    params, opt, _ = make_optimizer(model, cfg, 10**6, device)

    def timed(fn, n):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            train_step(model, cfg, opt, params, emb, act, device, loss_fn=fn)
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    timed(loss_fn, 3)  # échauffement
    t_eager, t_compiled = timed(loss_fn, 30), None
    try:
        timed(compiled, 3)  # la compilation a lieu ici
        t_compiled = timed(compiled, 30)
    except Exception as err:  # noqa: BLE001  (on veut juste savoir si ça marche)
        log(f"torch.compile ne marche pas ici ({type(err).__name__}) : on s'en passe")
    model.load_state_dict(state)
    torch.set_rng_state(rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    use = t_compiled is not None and t_compiled < 0.9 * t_eager
    compiled_txt = f"{t_compiled * 1e3:.1f} ms" if t_compiled is not None else "échec"
    log(f"torch.compile : {t_eager * 1e3:.1f} ms par pas sans, {compiled_txt} avec -> {'activé' if use else 'désactivé'}")
    return compiled if use else loss_fn


# ---------- Sauvegarde et reprise ----------


def save_state(path, model, opt, sched, generator, epoch, epochs):
    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "epoch": epoch,
        "epochs": epochs,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_state(path, model, opt, sched, generator, epochs):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["epochs"] != epochs:
        raise SystemExit(f"{path} vient d'un run de {state['epochs']} epochs : relance avec --epochs "
                         f"{state['epochs']}, ou ajoute --fresh pour repartir de zéro.")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["epoch"] + 1


def default_run_name(env, model, seed):
    if (env, model, seed) == ("tworoom", "joint_flow", 0):
        return "fjepa-tworoom-frozen"  # nom du premier test (RUN_NAME dans env.sh)
    return f"{model}-{env}-s{seed}"


def main():
    global LOG_FILE
    parser = argparse.ArgumentParser(description="Prédicteur sur encodeur LeWM gelé (Flow-JEPA, variantes, LeWM).")
    parser.add_argument("--env", choices=list(ENVS), default="tworoom")
    parser.add_argument("--model", choices=[*MODELS, "lewm"], default="joint_flow")
    parser.add_argument("--epochs", type=int, default=20, help="20 dans l'article")
    parser.add_argument("--seed", type=int, default=0, help="0 = graine de leur config ; 1, 2... = autres entraînements")
    parser.add_argument("--run-name", default=None, help="nom du dossier des poids (par défaut : modèle-env-graine)")
    parser.add_argument("--save-every", type=int, default=1, help="sauvegarder les poids toutes les N epochs")
    parser.add_argument("--compile", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--encode-only", action="store_true", help="calculer seulement les vecteurs, puis s'arrêter")
    parser.add_argument("--fresh", action="store_true", help="ignorer last.pt et repartir de zéro")
    args = parser.parse_args()
    run_name = args.run_name or default_run_name(args.env, args.model, args.seed)
    LOG_FILE = REPO_DIR / "logs" / f"train_{run_name}.log"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("ATTENTION : pas de GPU visible, l'entraînement sera très lent.")

    cfg = load_cfg(args.epochs, args.env, args.model)
    torch.manual_seed(cfg.seed + args.seed)
    dataset = load_dataset(cfg)
    model = build_model(cfg, dataset, device)
    embs = load_or_encode(model, dataset, device, emb_cache(args.env))
    if args.encode_only:
        return
    actions = normalized_actions(dataset, device)
    get_batch = make_batch_fn(dataset, embs, actions, device, token_dim=not is_lewm(cfg))

    # Même découpage train/val que leur train.py : même fonction, même graine, quelle que soit --seed.
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator
    )
    if args.seed:  # autre entraînement : autre ordre des exemples (et autre départ des poids, voir manual_seed)
        generator = torch.Generator().manual_seed(cfg.seed + args.seed)
    train_ids = torch.tensor(train_set.indices, device=device)
    val_ids = torch.tensor(val_set.indices, device=device)
    batch_size = cfg.loader.batch_size
    steps_per_epoch = len(train_ids) // batch_size  # drop_last=True, comme leur DataLoader
    loss_fn = choose_compile(args.compile, model, cfg, make_loss_fn(model, cfg), get_batch, train_ids, device)
    params, opt, sched = make_optimizer(model, cfg, steps_per_epoch * args.epochs, device)

    ckpt_dir = STABLEWM_HOME / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last, history_file = ckpt_dir / "last.pt", REPO_DIR / "logs" / f"history_{run_name}.json"
    start_epoch, history = 1, []
    if last.exists() and not args.fresh:
        start_epoch = load_state(last, model, opt, sched, generator, args.epochs)
        history = json.loads(history_file.read_text()) if history_file.exists() else []
        log(f"Reprise après l'epoch {start_epoch - 1} (fichier {last})")
    log(
        f"{run_name} : modèle {args.model}, {args.env}, graine {args.seed} | {fmt(len(dataset))} exemples "
        f"({fmt(len(train_ids))} train, {fmt(len(val_ids))} val), {fmt(steps_per_epoch)} pas par epoch, {args.epochs} epochs"
    )

    set_train_mode(model, cfg, True)
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        order = train_ids[torch.randperm(len(train_ids), generator=generator).to(device)]
        loss_sum = torch.zeros((), device=device)
        for s in range(steps_per_epoch):
            emb, act = get_batch(order[s * batch_size : (s + 1) * batch_size])
            loss_sum += train_step(model, cfg, opt, params, emb, act, device, sched, loss_fn=loss_fn)
        train_loss = loss_sum.item() / steps_per_epoch
        val_loss = validation_loss(model, cfg, get_batch, val_ids, device)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            swm.wm.utils.save_pretrained(
                model, run_name=run_name, config=cfg.model, filename=f"weights_epoch_{epoch}.pt",
                cache_dir=str(STABLEWM_HOME),
            )
        save_state(last, model, opt, sched, generator, epoch, args.epochs)
        minutes = (time.time() - t0) / 60
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": sched.get_last_lr()[0], "minutes": round(minutes, 2)})
        history_file.write_text(json.dumps(history, indent=1))
        remaining = (time.time() - start) / (epoch - start_epoch + 1) * (args.epochs - epoch)
        log(
            f"epoch {epoch:2d}/{args.epochs} | perte train {train_loss:.4f} | perte val {val_loss:.4f} | "
            f"lr {sched.get_last_lr()[0]:.1e} | {minutes:.1f} min | reste ~{remaining / 60:.0f} min"
        )
    log(f"Terminé en {(time.time() - start) / 60:.0f} min. Poids : {ckpt_dir}")


if __name__ == "__main__":
    main()
