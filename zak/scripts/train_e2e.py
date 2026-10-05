"""Le vrai test : Flow-JEPA sur TwoRoom, encodeur appris de zéro, comme dans l'article.
Aussi pour ses variantes (voir variants.py), avec la même recette :
  --model joint_flow  Flow-JEPA de l'article (5 pas d'un coup + flow matching), par défaut
  --model joint_det   Direct : 5 pas d'un coup, calculés en une passe (sans brouillon aléatoire)
  --model ar_flow     Flow pas à pas : chaque pas par flow matching
  --model ar_det      Direct pas à pas : chaque pas calculé en une passe (comme LeWM, avec leur réseau)
et trois configurations (--config) : tworoom (par défaut), reacher (réglages de l'article pour Reacher :
départ de l'état actuel, attention causale) et reacher_sans_causal (départ de l'état actuel seulement).
--seed change le tirage du départ du réseau et l'ordre des exemples, pas le découpage train/val.

Même calcul que leur train.py : leur config, leur modèle, leur dataset, leur découpage train/val,
et leur fonction de perte flow_jepa_forward (flow matching + SIGReg), appelée telle quelle.
Ce qui change, pour aller plus vite :
- les images voyagent en uint8 (4 fois moins de données) et sont normalisées sur le GPU ;
- un processus de chargement par vCPU disponible (leur config en utilise 6) ;
- optimiseur fusionné, et torch.compile si le benchmark montre qu'il marche et fait gagner du temps ;
- reprise automatique à la dernière epoch terminée après une coupure (fichier last.pt).

Usage :
  python scripts/train_e2e.py --benchmark          # 3 à 5 min : vitesse réelle, test de torch.compile, durée prévue
  python scripts/train_e2e.py [--epochs 20] [--compile] [--model joint_det] [--config reacher] [--seed 1]
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
from omegaconf import open_dict

import train_frozen_fast as tff  # chemins, chargement du dataset, autocast, optimiseur (vérifiés au test gelé)

import hydra  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from module import SIGReg  # noqa: E402  (leur module.py)
from train import flow_jepa_forward  # noqa: E402  (leur fonction de perte, telle quelle)
from utils import get_column_normalizer  # noqa: E402  (leur normalisation des actions)

LOG_FILE = tff.REPO_DIR / "logs" / "train_e2e.log"  # remplacé par un fichier par entraînement dans main()
BENCH_FILE = tff.REPO_DIR / "logs" / "benchmark.json"
VAL_SAMPLES = 8192  # validation sur un sous-ensemble fixe de la partie val : c'est un indicateur, pas l'entraînement
LOG_EVERY = 500  # pas entre deux lignes de progression


def log(msg):
    print(msg, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# ---------- Config, données, modèle : comme leur train.py ----------


# Réglages de l'article (annexe A). tworoom : config par défaut de leur repo. reacher : départ du flow
# autour de l'état actuel N(z_t, 0,5²) et attention causale entre positions futures (dropout 0,1 dans les deux).
CONFIGS = {
    "tworoom": [],
    "reacher": [
        "flow_matching.source=noisy_current",
        "flow_matching.source_noise_scale=0.5",
        "model.predictor.causal_self_attention=true",
    ],
    "reacher_sans_causal": [  # pour savoir lequel des deux réglages de « reacher » compte
        "flow_matching.source=noisy_current",
        "flow_matching.source_noise_scale=0.5",
    ],
}
# Fin du nom de l'entraînement, pour chaque configuration.
CONFIG_SUFFIX = {"tworoom": "", "reacher": "-reacherconfig", "reacher_sans_causal": "-reacherconfig-sanscausal"}


def default_run_name(model, config, seed=None):
    """Nom d'un entraînement. Flow-JEPA avec sa config par défaut garde le nom du premier run."""
    if (model, config) == ("joint_flow", "tworoom"):
        name = "fjepa-tworoom-full"
    else:
        name = f"{model}-tworoom-e2e{CONFIG_SUFFIX[config]}"
    return name if seed is None else f"{name}-s{seed}"


def load_cfg(epochs, model="joint_flow", config="tworoom"):
    """Leur config d'entraînement, réglée comme l'article pour TwoRoom (SIGReg 0,1), pour le modèle demandé."""
    overrides = ["data=tworoom", f"trainer.max_epochs={epochs}", "loss.sigreg.weight=0.1", *CONFIGS[config]]
    with initialize_config_dir(config_dir=str(tff.FJEPA_DIR / "config" / "train"), version_base=None):
        cfg = compose(config_name="fjepa", overrides=overrides)
    if model != "joint_flow":  # A garde leur classe JEPA telle quelle
        with open_dict(cfg):
            cfg.model._target_ = "variants.VariantJEPA"
            cfg.model.joint, cfg.model.flow = tff.MODELS[model]
    return cfg


def build_data(cfg):
    """Leur dataset et leur découpage train/val. Seule différence : les images restent en uint8
    (leur prétraitement des images est fait sur le GPU, voir gpu_batch)."""
    dataset = tff.load_dataset(cfg)
    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.data.dataset.keys_to_load
        if not col.startswith("pixels")
    ]
    dataset.transform = spt.data.transforms.Compose(*normalizers)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator
    )
    return dataset, train_set, val_set, generator


def build_model(cfg, dataset, device):
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = dataset.frameskip * dataset.get_dim("action")
    return hydra.utils.instantiate(cfg.model).to(device)


def num_workers(requested=0):
    """Un processus de chargement par vCPU, moins 2 (le processus principal et une marge).
    requested : nombre imposé (quand plusieurs entraînements se partagent le processeur)."""
    if requested:
        return requested
    cpus = int(os.environ.get("RUNPOD_CPU_COUNT", 0))  # nombre de vCPU du pod, fourni par RunPod
    if not cpus:
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(2, min(16, cpus - 2))


def make_loader(subset, batch_size, workers, shuffle, generator=None, persistent=True):
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # drop_last=True pour l'entraînement, comme leur DataLoader
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        prefetch_factor=4,
    )


def gpu_batch_fn(device):
    """Leur get_img_preprocessor (utils.py), fait sur le GPU : [0, 255] -> [0, 1] -> normalisation ImageNet.
    Leur Resize(224) ne change rien : les images font déjà 224 x 224."""
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, 1, 3, 1, 1)

    def gpu_batch(batch):
        pixels = batch["pixels"].to(device, non_blocking=True)  # (B, 6, 3, 224, 224) en uint8
        return {
            "pixels": (pixels.float() / 255 - mean) / std,
            "action": batch["action"].to(device, non_blocking=True),
        }

    return gpu_batch


# ---------- Perte : leur flow_jepa_forward, appelée telle quelle ----------


class LossModule:
    """Ce que leur flow_jepa_forward attend de « self » (normalement un module Lightning)."""

    def __init__(self, model, sigreg):
        self.model, self.sigreg = model, sigreg

    def log_dict(self, *args, **kwargs):
        pass  # les pertes sont journalisées par notre boucle


def make_loss_fn(model, sigreg, cfg, compile_model=False):
    # Seuls les 3 réglages lus par flow_jepa_forward, dans un objet simple (plus facile pour torch.compile).
    settings = SimpleNamespace(
        history_size=cfg.history_size,
        num_preds=cfg.num_preds,
        loss=SimpleNamespace(sigreg=SimpleNamespace(weight=cfg.loss.sigreg.weight)),
    )
    module = LossModule(model, sigreg)

    def loss_fn(batch):
        out = flow_jepa_forward(module, dict(batch), "train", settings)
        return out["loss"], out["pred_loss"], out["sigreg_loss"]

    return torch.compile(loss_fn) if compile_model else loss_fn


def train_step(loss_fn, batch, opt, sched, params, clip, device):
    with tff.autocast(device):
        loss, pred_loss, sigreg_loss = loss_fn(batch)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, clip)
    opt.step()
    sched.step()
    return torch.stack([loss.detach(), pred_loss.detach(), sigreg_loss.detach()]).float()


@torch.no_grad()
def validate(model, loss_fn, loader, gpu_batch, device):
    model.eval()
    total, count = None, 0
    for batch in loader:
        with tff.autocast(device):
            losses = torch.stack(loss_fn(gpu_batch(batch))).float()
        n = len(batch["action"])
        total = losses * n if total is None else total + losses * n
        count += n
    model.train()
    return (total / count).tolist()


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
    os.replace(tmp, path)  # écriture atomique : un fichier à moitié écrit ne remplace jamais le bon


def load_state(path, model, opt, sched, generator, epochs):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["epochs"] != epochs:
        raise SystemExit(
            f"{path} vient d'un run de {state['epochs']} epochs : relance avec --epochs {state['epochs']}, "
            "ou ajoute --fresh pour repartir de zéro."
        )
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["epoch"] + 1


# ---------- Benchmark : vitesse réelle avant de lancer des heures de calcul ----------


def timed_steps(loss_fn, batch, opt, sched, params, clip, device, n):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        train_step(loss_fn, batch, opt, sched, params, clip, device)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def benchmark(model, sigreg, cfg, train_loader, gpu_batch, device, epochs):
    batch_size, steps_per_epoch = cfg.loader.batch_size, len(train_loader)
    clip = cfg.trainer.gradient_clip_val

    # 1. Chargement des données par les processus du CPU
    it = iter(train_loader)
    for _ in range(3):  # démarrage des processus
        next(it)
    t0, n = time.time(), min(20, steps_per_epoch - 3)
    for _ in range(n):
        cpu_batch = next(it)
    load_speed = n * batch_size / (time.time() - t0)
    log(f"[1/3] chargement : {tff.fmt(int(load_speed))} exemples/s avec {train_loader.num_workers} processus")

    # 2. Pas d'entraînement sur le GPU, sans torch.compile
    batch = gpu_batch(cpu_batch)
    params, opt, sched = tff.make_optimizer(model, cfg, 10**6, device)
    eager_fn = make_loss_fn(model, sigreg, cfg)
    timed_steps(eager_fn, batch, opt, sched, params, clip, device, 3)  # échauffement
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_eager = timed_steps(eager_fn, batch, opt, sched, params, clip, device, 10)
    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    log(f"[2/3] pas d'entraînement : {t_eager * 1e3:.0f} ms ({batch_size / t_eager:.0f} exemples/s), mémoire GPU {mem:.1f} Go")

    # 3. Avec torch.compile (regroupe les opérations du GPU)
    t_compiled = None
    try:
        compiled_fn = make_loss_fn(model, sigreg, cfg, compile_model=True)
        t0 = time.time()
        timed_steps(compiled_fn, batch, opt, sched, params, clip, device, 3)  # la compilation a lieu ici
        compile_time = time.time() - t0
        t_compiled = timed_steps(compiled_fn, batch, opt, sched, params, clip, device, 10)
        log(f"[3/3] avec torch.compile : {t_compiled * 1e3:.0f} ms par pas (compilation : {compile_time:.0f} s)")
    except Exception as err:  # noqa: BLE001  (on veut juste savoir si ça marche)
        log(f"[3/3] torch.compile ne marche pas ici ({type(err).__name__}) : on s'en passera")

    use_compile = t_compiled is not None and t_compiled < 0.9 * t_eager
    step_time = max(t_compiled if use_compile else t_eager, batch_size / load_speed)
    hours = epochs * steps_per_epoch * step_time * 1.03 / 3600  # +3 % pour la validation
    bottleneck = "le chargement (CPU)" if batch_size / load_speed > step_time * 0.99 else "le GPU"
    log(f"Temps prévu : {hours:.1f} h pour {epochs} epochs ({tff.fmt(steps_per_epoch)} pas par epoch). Frein : {bottleneck}.")
    log(f"torch.compile : {'oui' if use_compile else 'non'} (choix enregistré dans {BENCH_FILE.name})")
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCH_FILE.write_text(json.dumps({"compile": use_compile, "hours": round(hours, 2), "memory_gb": round(mem, 1)}))


# ---------- Entraînement ----------


def main():
    global LOG_FILE
    parser = argparse.ArgumentParser(description="Flow-JEPA sur TwoRoom, encodeur appris (protocole de l'article).")
    parser.add_argument("--epochs", type=int, default=20, help="20 dans l'article")
    parser.add_argument("--model", choices=["joint_flow", "joint_det", "ar_flow", "ar_det"], default="joint_flow")
    parser.add_argument("--config", choices=list(CONFIGS), default="tworoom")
    parser.add_argument("--seed", type=int, default=None, help="graine du réseau (par défaut : celle de leur config)")
    parser.add_argument("--run-name", default=None, help="par défaut : voir default_run_name")
    parser.add_argument("--compile", action="store_true", help="utiliser torch.compile (conseillé par --benchmark ?)")
    parser.add_argument("--workers", type=int, default=0, help="processus de chargement (0 = automatique)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="pour les tests seulement")
    parser.add_argument("--benchmark", action="store_true", help="mesurer la vitesse et la durée prévue, sans entraîner")
    parser.add_argument("--fresh", action="store_true", help="ignorer last.pt et repartir de zéro")
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = default_run_name(args.model, args.config, args.seed)
    LOG_FILE = tff.REPO_DIR / "logs" / f"train_e2e_{args.run_name}.log"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("ATTENTION : pas de GPU visible, l'entraînement sera très lent.")

    cfg = load_cfg(args.epochs, args.model, args.config)
    torch.manual_seed(cfg.seed)
    dataset, train_set, val_set, generator = build_data(cfg)  # découpage train/val toujours avec leur graine
    if args.seed is not None:  # autre graine : autre départ du réseau, autre ordre des exemples
        torch.manual_seed(args.seed)
        generator.manual_seed(args.seed)
    model = build_model(cfg, dataset, device)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)
    gpu_batch = gpu_batch_fn(device)
    workers = num_workers(args.workers)
    train_loader = make_loader(train_set, cfg.loader.batch_size, workers, shuffle=True, generator=generator)

    if args.benchmark:
        benchmark(model, sigreg, cfg, train_loader, gpu_batch, device, args.epochs)
        return

    val_subset = torch.utils.data.Subset(val_set, range(min(VAL_SAMPLES, len(val_set))))
    steps_per_epoch = min(len(train_loader), args.max_steps_per_epoch or len(train_loader))
    params, opt, sched = tff.make_optimizer(model, cfg, steps_per_epoch * args.epochs, device)
    ckpt_dir = tff.STABLEWM_HOME / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pt"
    history_file = tff.REPO_DIR / "logs" / f"history_{args.run_name}.json"
    start_epoch, history = 1, []
    if last.exists() and not args.fresh:
        start_epoch = load_state(last, model, opt, sched, generator, args.epochs)
        history = json.loads(history_file.read_text()) if history_file.exists() else []
        log(f"Reprise après l'epoch {start_epoch - 1} (fichier {last})")
    train_fn = make_loss_fn(model, sigreg, cfg, compile_model=args.compile)
    eval_fn = make_loss_fn(model, sigreg, cfg)  # validation sans compilation : autre taille de batch, mode eval
    log(
        f"{args.run_name} : modèle {args.model}, config {args.config}, graine {args.seed or cfg.seed} | "
        f"{tff.fmt(len(dataset))} exemples ({tff.fmt(len(train_set))} train, {tff.fmt(len(val_subset))} val suivis), "
        f"{tff.fmt(steps_per_epoch)} pas par epoch, {args.epochs} epochs, {workers} processus de chargement, "
        f"torch.compile : {'oui' if args.compile else 'non'}"
    )

    model.train()
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t0, waited = time.time(), 0.0
        sums = torch.zeros(3, device=device)
        batches = iter(train_loader)
        for step in range(1, steps_per_epoch + 1):
            tw = time.time()
            cpu_batch = next(batches)
            waited += time.time() - tw
            batch = gpu_batch(cpu_batch)
            try:
                sums += train_step(train_fn, batch, opt, sched, params, cfg.trainer.gradient_clip_val, device)
            except Exception as err:  # noqa: BLE001  (torch.compile qui échoue : on continue sans)
                if train_fn is eval_fn:
                    raise
                log(f"torch.compile a échoué ({type(err).__name__}) : on continue sans")
                train_fn = eval_fn
                sums += train_step(train_fn, batch, opt, sched, params, cfg.trainer.gradient_clip_val, device)
            if step % LOG_EVERY == 0:
                loss, pred, sig = (sums / step).tolist()
                elapsed = time.time() - t0
                log(
                    f"  epoch {epoch} pas {tff.fmt(step)}/{tff.fmt(steps_per_epoch)} | perte {loss:.4f} "
                    f"(prédiction {pred:.4f}, sigreg {sig:.4f}) | {step * cfg.loader.batch_size / elapsed:.0f} ex/s "
                    f"| attente des données {waited / elapsed:.0%}"
                )
        loss, pred, sig = (sums / steps_per_epoch).tolist()
        # Même taille de batch qu'à l'entraînement : la valeur de SIGReg est proportionnelle à la taille du batch.
        val_loader = make_loader(val_subset, cfg.loader.batch_size, max(2, workers // 2), shuffle=False, persistent=False)
        v_loss, v_pred, v_sig = validate(model, eval_fn, val_loader, gpu_batch, device)
        swm.wm.utils.save_pretrained(
            model, run_name=args.run_name, config=cfg.model, filename=f"weights_epoch_{epoch}.pt",
            cache_dir=str(tff.STABLEWM_HOME),
        )
        save_state(last, model, opt, sched, generator, epoch, args.epochs)
        history.append({"epoch": epoch, "train_loss": pred, "val_loss": v_pred, "train_total": loss, "val_total": v_loss,
                        "sigreg_train": sig, "sigreg_val": v_sig, "lr": sched.get_last_lr()[0],
                        "minutes": round((time.time() - t0) / 60, 2)})
        history_file.write_text(json.dumps(history, indent=1))
        done = epoch - start_epoch + 1
        remaining = (time.time() - start) / done * (args.epochs - epoch)
        log(
            f"epoch {epoch:2d}/{args.epochs} | train : perte {loss:.4f} (prédiction {pred:.4f}, sigreg {sig:.4f}) "
            f"| val : perte {v_loss:.4f} (prédiction {v_pred:.4f}, sigreg {v_sig:.4f}) | lr {sched.get_last_lr()[0]:.1e} "
            f"| {(time.time() - t0) / 60:.1f} min | reste ~{remaining / 3600:.1f} h"
        )
    log(f"Terminé en {(time.time() - start) / 3600:.1f} h. Poids : {ckpt_dir}")


if __name__ == "__main__":
    main()
