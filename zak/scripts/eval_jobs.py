"""File d'évaluations en parallèle, réparties sur les GPU et les cœurs du processeur.

Chaque évaluation lance leur eval.py (eval_lewm.py pour les modèles LeWM) avec le protocole de l'article.
Les évaluations les plus longues partent en premier. À la fin : summary.md et summary.json.

Pour chaque modèle entraîné (--runs) et pour LeWM officiel :
  - protocole de l'article : 50 épisodes, images propres et bruitées, planificateur à 10 et à 30 itérations ;
  - plus précis : 200 épisodes, propres et bruitées, 10 itérations ;
  - plateau : poids de l'epoch 10, 50 épisodes, propres et bruitées, 10 itérations.
Plus la politique aléatoire (50 et 200 épisodes).

Usage :
  python scripts/eval_jobs.py --env pusht --exp pusht-chain --runs lewm-pusht-s0 joint_flow-pusht-s0 ...
  python scripts/eval_jobs.py --env tworoom --exp X --runs mon-run --no-baselines   # seulement ce modèle
  python scripts/eval_jobs.py --env tworoom --exp X --summary-only   # tableau de toutes les évaluations de X
  (options Hydra en plus pour toutes les évaluations, ex. sur Mac : --extra solver.device=cpu)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
FJEPA_DIR = REPO_DIR / "third_party" / "Flow-JEPA"
STABLEWM_HOME = Path(os.environ.get("STABLEWM_HOME", "/workspace/stable-wm"))
ENVS = {  # rayon de la tache de bruit et fichier de résultats de leur eval.py, pour chaque environnement
    "tworoom": {"radius": 35, "results": "tworoom_results.txt", "lewm": "lewm-tworooms"},
    "pusht": {"radius": 10, "results": "pusht_results.txt", "lewm": "lewm-pusht"},
}
DESCRIPTIONS = {
    "lewm-officiel": "LeWM officiel (référence)",
    "lewm": "LeWM réentraîné par nous",
    "ar_det": "Direct pas à pas (réseau Flow-JEPA, sans flow matching)",
    "ar_flow": "Flow pas à pas",
    "joint_det": "Direct, 5 pas d'un coup (sans flow matching)",
    "joint_flow": "Flow-JEPA de l'article (5 pas d'un coup + flow matching)",
    "fjepa": "Flow-JEPA de l'article (5 pas d'un coup + flow matching)",
    "random": "actions au hasard",
}
SLOWNESS = {"ar_flow": 5.0, "joint_flow": 1.5}  # coût relatif de la planification, pour l'ordre de passage
RUNNING = set()  # évaluations en cours, pour pouvoir les arrêter (run_variants.py)


def kind_of(run):
    return next((k for k in DESCRIPTIONS if run == k or run.startswith(k + "-")), run)


def describe(run):
    text = DESCRIPTIONS.get(kind_of(run), "")
    if "-reacherconfig-sanscausal" in run:
        text += " ; départ de l'état actuel, sans attention causale"
    elif "-reacherconfig" in run:
        text += " ; départ de l'état actuel + attention causale (réglages Reacher)"
    seed = re.search(r"-s(\d+)$", run)
    if seed:
        text += f" ; graine {seed.group(1)}"
    return text


def build_jobs(args):
    jobs = []

    def add(label, policy, script, epoch, cem, noisy, episodes):
        name = f"{label}_e{epoch}_cem{cem}_{'bruit' if noisy else 'propre'}_ep{episodes}"
        cost = episodes * (1 + cem / 10 * SLOWNESS.get(kind_of(label), 0.5))
        jobs.append(dict(name=name, label=label, policy=str(policy), script=script, epoch=epoch,
                         cem=cem, noisy=noisy, episodes=episodes, cost=cost))

    def add_model(label, policy, script, epoch):
        for cem in args.cems:
            for noisy in (False, True):
                add(label, policy, script, epoch, cem, noisy, args.episodes)
        for noisy in (False, True):
            add(label, policy, script, epoch, args.cems[0], noisy, args.precise_episodes)

    if not args.no_baselines:
        for n in (args.episodes, args.precise_episodes):
            add("random", "random", "eval.py", 0, args.cems[0], False, n)
        lewm_official = STABLEWM_HOME / "checkpoints" / ENVS[args.env]["lewm"]
        add_model("lewm-officiel", lewm_official, "eval_lewm.py", 0)
    for run in args.runs:
        folder = STABLEWM_HOME / "checkpoints" / run
        # Pas de config.json : modèle entraîné sur un autre pod (ses évaluations sont déjà faites)
        target = json.loads((folder / "config.json").read_text())["_target_"] if (folder / "config.json").exists() else ""
        script = "eval_lewm.py" if target.endswith("LeWM") else "eval.py"
        add_model(run, folder / f"weights_epoch_{args.final_epoch}.pt", script, args.final_epoch)
        if args.check_epoch and (folder / f"weights_epoch_{args.check_epoch}.pt").exists():
            for noisy in (False, True):
                add(run, folder / f"weights_epoch_{args.check_epoch}.pt", script, args.check_epoch,
                    args.cems[0], noisy, args.episodes)
    return sorted(jobs, key=lambda j: -j["cost"])


def visible_gpus():
    """Numéros réels des GPU visibles. Si ce processus est déjà limité à certains GPU (CUDA_VISIBLE_DEVICES),
    ses évaluations doivent recevoir ces numéros-là, pas 0, 1, 2..."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]


def run_job(job, out_dir, gpu, extra, env):
    job_dir = out_dir / job["name"]
    job_dir.mkdir(parents=True, exist_ok=True)
    script = FJEPA_DIR / "eval.py" if job["script"] == "eval.py" else REPO_DIR / "scripts" / "eval_lewm.py"
    cmd = [
        sys.executable, str(script), f"--config-path={FJEPA_DIR / 'config' / 'eval'}", f"--config-name={env}",
        f"policy={job['policy']}", f"perturbation.enabled={str(job['noisy']).lower()}",
        f"perturbation.gaussian_noise.radius={ENVS[env]['radius']}", f"solver.n_steps={job['cem']}",
        f"eval.num_eval={job['episodes']}", "output.save_video=false",
        f"output.dir={job_dir}", f"hydra.run.dir={job_dir / 'hydra'}", *extra,
    ]
    child_env = dict(os.environ, OMP_NUM_THREADS="2",
                     PYTHONPATH=os.pathsep.join([str(REPO_DIR / "scripts"), os.environ.get("PYTHONPATH", "")]))
    if gpu is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = visible_gpus()[gpu]
    t0 = time.time()
    with open(out_dir / f"{job['name']}.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=FJEPA_DIR, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        RUNNING.add(proc)
        code = proc.wait()
        RUNNING.discard(proc)
    result = dict(job, minutes=round((time.time() - t0) / 60, 1), exit_code=code, success_rate=None, successes=None)
    results_file = job_dir / ENVS[env]["results"]
    if code == 0 and results_file.exists():
        text = results_file.read_text()
        rate = re.findall(r"'success_rate': ([0-9.]+)", text)
        flags = re.findall(r"'episode_successes': array\(\[(.*?)\]", text, re.S)
        result["success_rate"] = float(rate[-1]) if rate else None
        result["successes"] = [w == "True" for w in re.findall(r"True|False", flags[-1])] if flags else None
    (job_dir / "job.json").write_text(json.dumps(result))  # pour reconstruire le tableau complet (--summary-only)
    return result


def write_summary(results, out_dir, env):
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    rows = {}
    for r in results:
        key = (r["label"], r["epoch"], r["episodes"], r["cem"])
        rows.setdefault(key, {})["bruit" if r["noisy"] else "propre"] = r["success_rate"]
    lines = [f"# Résultats {env} (taux de succès en %)", "",
             "| modèle | description | epoch | épisodes | itérations CEM | propre | bruité |",
             "|---|---|---|---|---|---|---|"]
    order = list(DESCRIPTIONS)
    rank = lambda label: order.index(kind_of(label)) if kind_of(label) in order else len(order)  # noqa: E731
    for (label, epoch, episodes, cem), v in sorted(rows.items(), key=lambda kv: (rank(kv[0][0]), kv[0])):
        show = lambda x: "—" if x is None else f"{x:.1f}"  # noqa: E731
        lines.append(f"| {label} | {describe(label)} | {epoch or ''} | {episodes} | {cem} "
                     f"| {show(v.get('propre'))} | {show(v.get('bruit'))} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Évaluations en parallèle (protocole de l'article).")
    parser.add_argument("--env", choices=list(ENVS), default="pusht")
    parser.add_argument("--exp", required=True, help="nom de l'expérience : résultats dans results/<exp>/")
    parser.add_argument("--runs", nargs="*", default=[], help="noms des entraînements à évaluer")
    parser.add_argument("--final-epoch", type=int, default=20)
    parser.add_argument("--check-epoch", type=int, default=10, help="0 pour ne pas évaluer d'epoch intermédiaire")
    parser.add_argument("--episodes", type=int, default=50, help="protocole de l'article")
    parser.add_argument("--precise-episodes", type=int, default=200)
    parser.add_argument("--cems", type=int, nargs="+", default=[10, 30], help="itérations du planificateur")
    parser.add_argument("--parallel", type=int, default=0, help="évaluations simultanées (0 = automatique)")
    parser.add_argument("--no-baselines", action="store_true", help="ne pas évaluer LeWM officiel ni le hasard")
    parser.add_argument("--summary-only", action="store_true", help="refaire le tableau de toutes les évaluations faites")
    parser.add_argument("--extra", nargs="*", default=[], help="options Hydra ajoutées à toutes les évaluations")
    args = parser.parse_args()

    out_dir = REPO_DIR / "results" / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        results = [json.loads(f.read_text()) for f in sorted(out_dir.glob("*/job.json"))]
        write_summary(results, out_dir, args.env)
        missing = [r["name"] for r in results if r["success_rate"] is None]
        sys.exit(1 if missing or not results else 0)
    jobs = build_jobs(args)
    n_gpus = torch.cuda.device_count()
    cpus = int(os.environ.get("RUNPOD_CPU_COUNT", 0)) or os.cpu_count()
    parallel = args.parallel or max(1, min(cpus - 2, 6 * max(1, n_gpus)))
    print(f"{len(jobs)} évaluations, {parallel} en même temps, sur {n_gpus} GPU", flush=True)

    load = [0] * n_gpus  # évaluations en cours sur chaque GPU
    lock = threading.Lock()
    results, done = [], [0]

    def worker(job):
        with lock:
            gpu = min(range(n_gpus), key=lambda g: load[g]) if n_gpus else None
            if gpu is not None:
                load[gpu] += 1
        try:
            result = run_job(job, out_dir, gpu, args.extra, args.env)
        finally:
            if gpu is not None:
                with lock:
                    load[gpu] -= 1
        with lock:
            results.append(result)
            done[0] += 1
            rate = "ÉCHEC" if result["success_rate"] is None else f"{result['success_rate']:.1f} %"
            print(f"[{done[0]}/{len(jobs)}] {job['name']} : {rate} ({result['minutes']} min)", flush=True)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(worker, jobs))
    write_summary(results, out_dir, args.env)
    failed = [r["name"] for r in results if r["success_rate"] is None]
    if failed:
        print(f"{len(failed)} évaluation(s) en échec (voir results/{args.exp}/<nom>.log) : {failed}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
