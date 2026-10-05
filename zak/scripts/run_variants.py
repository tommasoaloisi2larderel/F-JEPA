"""Expérience TwoRoom, encodeur appris de zéro : variantes de Flow-JEPA (modèles et réglages : voir train_e2e.py).

TRAININGS liste tous les entraînements de l'expérience. Ceux dont les évaluations sont déjà dans
runs/tworoom-variants/summary.json (faites sur un pod précédent) ne sont pas refaits : pour ajouter
un entraînement, on ajoute une ligne et on relance. Le tableau final contient tout.

Ce programme répartit tout le travail sur les GPU du pod, pour qu'aucune carte n'attende :
  - un entraînement par GPU (train_e2e.py). S'il y a plus d'entraînements que de GPU,
    les suivants prennent la première carte libérée ;
  - dès qu'un entraînement est fini, ses évaluations entrent dans une file d'attente.
    Une carte sans entraînement fait tourner plusieurs évaluations à la fois ;
  - les références (LeWM officiel, hasard, Flow-JEPA du premier run) sont dans la même file dès le départ.
Évaluations (eval_jobs.py) : protocole de l'article (50 épisodes, CEM 30), 200 épisodes, poids de l'epoch 10.
À la fin, même après un échec : tableau des résultats, courbes de perte et logs
dans runs/tworoom-variants/ (c'est ce dossier que run_tworoom_variants.sh publie sur GitHub).
Modèles gardés : seulement Flow-JEPA, la référence (92 Mo par graine). Les variantes sont des
modèles de test, inutiles une fois évaluées.

Reprise après une coupure : relancer la même commande. Les entraînements repartent de leur
dernière epoch, les évaluations déjà réussies ne sont pas refaites.

Usage : python scripts/run_variants.py [--epochs 20]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import eval_jobs as ej
import plot_runs

REPO_DIR = ej.REPO_DIR
EXP = "tworoom-variants"
OUT = REPO_DIR / "runs" / EXP  # ce qui sera publié sur GitHub
RESULTS = REPO_DIR / "results" / EXP  # sorties des évaluations
LOGS = REPO_DIR / "logs"
LOG_FILE = LOGS / "run_variants.log"
# (modèle, réglages, graine) de chaque entraînement. Graine None : celle de leur config.
TRAININGS = [
    # Pod du 2026-10-03
    ("joint_det", "tworoom", None),  # Direct
    ("ar_flow", "tworoom", None),  # Flow pas à pas
    ("joint_det", "reacher", None),  # Direct depuis l'état actuel
    ("ar_flow", "reacher", None),  # Flow pas à pas depuis l'état actuel
    # Pod suivant : confirmer avec une 2e graine, et séparer les réglages de « Direct depuis l'état actuel »
    ("joint_det", "reacher", 1),  # Direct depuis l'état actuel, 2e graine
    ("joint_flow", "tworoom", 1),  # Flow-JEPA de l'article, 2e graine
    ("joint_det", "reacher_sans_causal", None),  # Direct depuis l'état actuel, sans attention causale
    ("ar_det", "reacher", None),  # Direct pas à pas depuis l'état actuel (l'attention causale n'y change rien)
]
REFERENCE = "fjepa-tworoom-full"  # Flow-JEPA du premier run (20 epochs), modèle dans runs/fjepa-tworoom-full/model
REFERENCE_EPOCH = 20
CONFIG_SUFFIX = {"tworoom": "", "reacher": "-reacherconfig", "reacher_sans_causal": "-reacherconfig-sanscausal"}


def run_name(model, config, seed):
    """Le nom que train_e2e.py donne par défaut à cet entraînement (voir default_run_name)."""
    name = "fjepa-tworoom-full" if (model, config) == ("joint_flow", "tworoom") else f"{model}-tworoom-e2e{CONFIG_SUFFIX[config]}"
    return name if seed is None else f"{name}-s{seed}"


def log(msg):
    line = f"[{time.strftime('%H:%M')}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def gpu_label(slot):
    return "CPU" if slot is None else f"GPU {ej.visible_gpus()[slot]}"


# ---------- Ce qu'il y a à faire ----------


def import_previous():
    """Reprend les résultats et les courbes déjà publiés (runs/tworoom-variants/), faits sur un pod précédent :
    leurs évaluations comptent comme faites, et leurs courbes restent dans courbes.png."""
    previous = OUT / "summary.json"
    if previous.exists():
        for result in json.loads(previous.read_text()):
            path = RESULTS / result["name"] / "job.json"
            if result["success_rate"] is not None and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result))
    for history in (OUT / "train_logs").glob("history_*.json"):
        if not (LOGS / history.name).exists():
            LOGS.mkdir(parents=True, exist_ok=True)
            shutil.copy(history, LOGS / history.name)


def to_train(args):
    """Les entraînements dont il manque au moins une évaluation finale."""
    return [t for t in TRAININGS if eval_jobs_for([run_name(*t)], args.epochs, False, args, final_only=True)]


def prepare_reference():
    """Copie le modèle Flow-JEPA du premier run là où les évaluations cherchent les modèles. False s'il manque."""
    src = REPO_DIR / "runs" / REFERENCE / "model"
    dst = ej.STABLEWM_HOME / "checkpoints" / REFERENCE
    weights = f"weights_epoch_{REFERENCE_EPOCH}.pt"
    if not (src / weights).exists():
        log(f"ATTENTION : {src / weights} introuvable, Flow-JEPA du premier run ne sera pas réévalué")
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for name in (weights, "config.json"):
        if not (dst / name).exists():
            shutil.copy(src / name, dst / name)
    return True


def eval_jobs_for(runs, final_epoch, baselines, args, final_only=False):
    """Les évaluations de ces modèles (définies dans eval_jobs.py), sauf celles déjà réussies.
    final_only : seulement celles des poids finaux (pas l'epoch intermédiaire)."""
    spec = SimpleNamespace(env="tworoom", runs=runs, final_epoch=final_epoch,
                           check_epoch=0 if final_only else args.check_epoch, no_baselines=not baselines,
                           episodes=args.episodes, precise_episodes=args.precise_episodes, cems=args.cems)
    return [job for job in ej.build_jobs(spec) if not already_done(job)]


def already_done(job):
    path = RESULTS / job["name"] / "job.json"
    return path.exists() and json.loads(path.read_text())["success_rate"] is not None


def start_training(model, config, seed, slot, workers, args):
    run = run_name(model, config, seed)
    cmd = [sys.executable, str(REPO_DIR / "scripts" / "train_e2e.py"), "--model", model, "--config", config,
           "--run-name", run, "--epochs", str(args.epochs), "--workers", str(workers)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if args.compile:
        cmd.append("--compile")
    if args.max_steps_per_epoch:
        cmd += ["--max-steps-per-epoch", str(args.max_steps_per_epoch)]
    # Les processus de compilation de torch.compile prennent aussi leur part du CPU, pas plus
    # (sinon jusqu'à 32 chacun, pour chacun des entraînements lancés en même temps).
    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS=str(workers))
    if slot is not None:
        env["CUDA_VISIBLE_DEVICES"] = ej.visible_gpus()[slot]
    with open(LOGS / f"stdout_{run}.log", "a") as out:  # erreurs Python éventuelles (le suivi est dans train_e2e_<run>.log)
        proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, stdout=out, stderr=subprocess.STDOUT)
    log(f"{gpu_label(slot)} : entraînement {run} lancé ({workers} processus de chargement)   "
        f"suivre : tail -f logs/train_e2e_{run}.log")
    return run, proc


# ---------- La répartition sur les GPU ----------


def stop_everything(training):
    """Arrête les entraînements et les évaluations en cours."""
    for _, proc in training.values():
        proc.terminate()
    for proc in list(ej.RUNNING):
        proc.terminate()


def schedule(args):
    """Lance tout et attend la fin. Renvoie la liste de ce qui a échoué."""
    slots = list(range(len(ej.visible_gpus()))) or [None]  # None : pas de GPU (test sur Mac)
    cpus = int(os.environ.get("RUNPOD_CPU_COUNT", 0)) or os.cpu_count()
    share = max(2, min(16, (cpus - 2) // len(slots)))  # vCPU par carte : chaque carte a la même part du processeur
    evals_per_gpu = args.evals_per_gpu or min(6, share)
    import_previous()
    pending = to_train(args)  # entraînements pas encore lancés
    for t in TRAININGS:
        if t not in pending:
            log(f"{run_name(*t)} : déjà évalué (runs/{EXP}/summary.json), pas réentraîné")
    rounds = math.ceil(len(pending) / len(slots))
    max_hours = args.max_hours or 4 + 8 * rounds  # limite de sécurité, large : 12 h avec 4 GPU
    log(f"{len(pending)} entraînements sur {len(slots)} carte(s), {cpus} vCPU : {share} processus de chargement "
        f"par entraînement, jusqu'à {evals_per_gpu} évaluations par carte libre. Limite de sécurité : {max_hours} h.")

    training = {}  # carte -> (nom du run, processus)
    evals = {slot: 0 for slot in slots}  # nombre d'évaluations en cours sur chaque carte
    finished = []  # résultats d'évaluations, déposés par les threads
    failures = []
    lock = threading.Lock()

    queue = eval_jobs_for([REFERENCE] if prepare_reference() else [], REFERENCE_EPOCH, True, args)
    log(f"{len(queue)} évaluations de référence (LeWM officiel, hasard, Flow-JEPA du premier run) à faire")

    def evaluate(job, slot):
        try:
            result = ej.run_job(job, RESULTS, slot, args.extra, "tworoom")
        except Exception as err:  # noqa: BLE001  (compté comme un échec, sans bloquer la boucle)
            result = dict(job, success_rate=None, minutes=0, error=repr(err))
        with lock:
            evals[slot] -= 1
            finished.append(result)

    deadline = time.time() + max_hours * 3600
    try:
        # « finished » aussi : le résultat de la dernière évaluation doit être lu (et réessayé s'il a échoué)
        while pending or training or queue or finished or any(evals.values()):
            # 1. Évaluations terminées : on garde le résultat, ou on réessaie une fois
            with lock:
                results, finished[:] = list(finished), []
            for r in results:
                if r["success_rate"] is not None:
                    log(f"évaluation {r['name']} : {r['success_rate']:.1f} % ({r['minutes']} min)")
                elif not r.get("retried"):
                    log(f"évaluation {r['name']} : ÉCHEC, on réessaie une fois (log : results/{EXP}/{r['name']}.log)")
                    queue.append(dict(r, retried=True))
                else:
                    log(f"évaluation {r['name']} : ÉCHEC à nouveau (log : results/{EXP}/{r['name']}.log)")
                    failures.append(r["name"])

            # 2. Entraînements terminés : leurs évaluations entrent dans la file
            for slot, (run, proc) in list(training.items()):
                code = proc.poll()
                if code is None:
                    continue
                del training[slot]
                if code == 0:
                    jobs = eval_jobs_for([run], args.epochs, False, args)
                    queue += jobs
                    log(f"{gpu_label(slot)} : entraînement {run} terminé, {len(jobs)} évaluations ajoutées à la file")
                else:
                    failures.append(run)
                    log(f"{gpu_label(slot)} : ÉCHEC de l'entraînement {run} (code {code}), "
                        f"voir logs/stdout_{run}.log et logs/train_e2e_{run}.log")

            # 3. Cartes libres : d'abord les entraînements (le plus long), puis les évaluations, les plus longues d'abord
            for slot in slots:
                if pending and slot not in training and evals[slot] == 0:
                    model, config, seed = pending.pop(0)
                    training[slot] = start_training(model, config, seed, slot, share, args)
            queue.sort(key=lambda job: -job["cost"])
            while queue and not pending:
                free = [s for s in slots if s not in training and evals[s] < evals_per_gpu]
                if not free:
                    break
                slot = min(free, key=lambda s: evals[s])  # la carte la moins chargée
                job = queue.pop(0)
                with lock:
                    evals[slot] += 1
                threading.Thread(target=evaluate, args=(job, slot), daemon=True).start()

            # 4. Sécurité : si quelque chose reste bloqué, on arrête tout plutôt que de payer des GPU pour rien
            if time.time() > deadline:
                log(f"LIMITE DE {max_hours} h DÉPASSÉE : arrêt de tout ce qui tourne encore")
                failures += [run for run, _ in training.values()] + ["limite de temps dépassée"]
                stop_everything(training)
                return failures
            time.sleep(args.poll)
    except BaseException:  # erreur ou Ctrl+C : on n'abandonne pas des processus qui tournent encore
        stop_everything(training)
        raise
    return failures


# ---------- Ce qu'on garde ----------


def collect(args):
    """Modèles Flow-JEPA, logs, tableau des résultats et courbes dans runs/tworoom-variants/.
    Les modèles et les logs d'abord : une erreur dans le tableau ou les courbes ne doit pas les perdre."""
    runs = [run_name(*t) for t in TRAININGS]
    (OUT / "train_logs").mkdir(exist_ok=True)  # pas « logs » : ce nom est ignoré par git (.gitignore)
    for run in runs:
        weights = ej.STABLEWM_HOME / "checkpoints" / run / f"weights_epoch_{args.epochs}.pt"
        if run.startswith("fjepa-") and weights.exists():  # Flow-JEPA : la référence, on la garde
            (OUT / "models" / run).mkdir(parents=True, exist_ok=True)
            shutil.copy(weights, OUT / "models" / run / weights.name)
            shutil.copy(weights.parent / "config.json", OUT / "models" / run / "config.json")
        for name in (f"train_e2e_{run}.log", f"history_{run}.json", f"stdout_{run}.log"):
            if (LOGS / name).exists():
                shutil.copy(LOGS / name, OUT / "train_logs" / name)
    try:
        results = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("*/job.json"))]
        if results:
            ej.write_summary(results, RESULTS, "tworoom")
            for name in ("summary.md", "summary.json"):
                shutil.copy(RESULTS / name, OUT / name)
        plot_runs.plot(runs, OUT / "courbes.png")
    except (Exception, SystemExit) as err:  # noqa: BLE001  (SystemExit : aucun historique de perte)
        log(f"tableau ou courbes incomplets : {err}")
    shutil.copy(LOG_FILE, OUT / "run_variants.log")


def main():
    parser = argparse.ArgumentParser(description="TwoRoom, encodeur appris : B et C, deux réglages chacune.")
    parser.add_argument("--epochs", type=int, default=20, help="20 comme A et l'article")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="sans torch.compile (tests sur Mac)")
    parser.add_argument("--evals-per-gpu", type=int, default=0, help="évaluations simultanées par carte (0 = selon les vCPU)")
    parser.add_argument("--max-hours", type=float, default=0, help="limite de sécurité (0 = 4 h + 8 h par vague d'entraînements)")
    parser.add_argument("--poll", type=float, default=20, help="secondes entre deux vérifications")
    # Pour les tests sur Mac : entraînements et évaluations minuscules.
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--check-epoch", type=int, default=10, help="epoch intermédiaire évaluée (0 = aucune)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--precise-episodes", type=int, default=200)
    parser.add_argument("--cems", type=int, nargs="+", default=[30], help="itérations du planificateur (30 sur TwoRoom)")
    parser.add_argument("--extra", nargs="*", default=[], help="options Hydra ajoutées à toutes les évaluations")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        failures = schedule(args)
    finally:  # même après une erreur : on garde tout ce qui existe
        collect(args)
    log(f"Fini en {(time.time() - start) / 3600:.1f} h. Résultats : runs/{EXP}/summary.md")
    if failures:
        log(f"Échecs : {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
