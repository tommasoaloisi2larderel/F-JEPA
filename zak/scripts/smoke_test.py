"""Test rapide (1 à 2 min), à lancer après setup.sh et avant l'entraînement.

1. le GPU est visible par PyTorch ;
2. le modèle se construit avec leur config, l'encodeur LeWM est chargé et gelé ;
3. le modèle apprend : sur un petit batch fixe, la perte de flow baisse ;
4. vitesse réelle de cette machine -> temps prévu pour train_frozen_fast.py.

Usage : python scripts/smoke_test.py [--epochs 20]
"""

import argparse
import sys
import time

import h5py
import torch

import train_frozen_fast as tff


def timed(fn, device, repeat):
    """Durée moyenne d'un appel, après un appel d'échauffement."""
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / repeat


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

# 1. GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/4] device = {device} ({torch.cuda.get_device_name() if device == 'cuda' else 'pas de GPU'})")
if device == "cpu":
    print("      ATTENTION : pas de GPU visible. L'entraînement sera inutilisable.")

# 2. Modèle : leur config + encodeur LeWM gelé
cfg = tff.load_cfg(args.epochs)
dataset = tff.load_dataset(cfg)
model = tff.build_model(cfg, dataset, device)
assert not any(p.requires_grad for p in model.encoder.parameters()), "encodeur non gelé"
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[2/4] encodeur LeWM chargé et gelé, {n_train / 1e6:.1f} M paramètres entraînables")

# 3. Le modèle apprend : il doit « par cœur » un petit batch fixe de vecteurs aléatoires.
dim, act_dim, steps = cfg.embed_dim, cfg.model.action_encoder.input_dim, dataset.num_steps
emb = torch.randn(16, steps, 1, dim, device=device)
act = torch.randn(16, steps, act_dim, device=device)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-4)
losses = [tff.train_step(model, cfg, opt, params, emb, act, device).item() for _ in range(100)]
# La perte de flow est bruitée (t et le bruit sont tirés au hasard) : on compare des moyennes.
first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
print(f"[3/4] perte de flow : {first:.3f} (début) -> {last:.3f} (fin)")
if last >= 0.7 * first:
    print("ÉCHEC : la perte ne baisse pas assez. Ne lance pas l'entraînement.")
    sys.exit(1)

# 4. Vitesse réelle -> temps prévu
block = tff.ENCODE_BLOCK if device == "cuda" else 16
pixels = torch.randint(0, 256, (block, 224, 224, 3), dtype=torch.uint8)
gpu_speed = block / timed(lambda: tff.encode_pixels(model, pixels, device), device, repeat=5)
with h5py.File(dataset.h5_path, "r") as f:
    n_frames = f["pixels"].shape[0]
    # Décompression d'un lot réel par un processus de lecture (le fichier en utilise READ_WORKERS).
    read_speed = tff.READ_WORKERS * block / timed(lambda: f["pixels"][:block], "cpu", repeat=3)
batch = cfg.loader.batch_size
emb = torch.randn(batch, steps, 1, dim, device=device)
act = torch.randn(batch, steps, act_dim, device=device)
params, opt, sched = tff.make_optimizer(model, cfg, 10**6, device)
step_time = timed(lambda: tff.train_step(model, cfg, opt, params, emb, act, device, sched), device, repeat=50)

encode_speed = min(gpu_speed, read_speed)
encode_min = 0 if tff.EMB_CACHE.exists() else n_frames / encode_speed / 60
steps_per_epoch = int(len(dataset) * cfg.train_split) // batch
train_min = args.epochs * steps_per_epoch * step_time * 1.05 / 60  # +5 % pour la validation
print(
    f"[4/4] encodeur : {tff.fmt(int(gpu_speed))} images/s ({device}), lecture : {tff.fmt(int(read_speed))} images/s, "
    f"pas d'entraînement : {step_time * 1e3:.1f} ms"
)
print(
    f"Temps prévu pour train_frozen_fast.py : encodage {encode_min:.0f} min "
    f"+ entraînement {train_min:.0f} min ({args.epochs} epochs)"
)
print("OK : tu peux lancer python scripts/train_frozen_fast.py")
