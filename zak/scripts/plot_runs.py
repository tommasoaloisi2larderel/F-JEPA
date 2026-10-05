"""Courbes d'apprentissage : la perte à chaque epoch, en entraînement et en validation, pour chaque modèle.

Un petit graphique par modèle, chacun avec sa propre échelle : la perte n'a pas la même définition
d'un modèle à l'autre (flow matching, erreur directe, perte de LeWM), on ne les compare donc pas entre eux.
À lire avec prudence : le taux d'apprentissage descend jusqu'à 0 à la dernière epoch, donc les courbes
s'aplatissent toujours à la fin. Pour savoir si plus d'epochs aiderait, regarder si la perte de
validation baissait encore au milieu, et comparer les taux de succès aux epochs 10 et 20.

Usage : python scripts/plot_runs.py --runs lewm-pusht-s0 joint_flow-pusht-s0 ... --out runs/pusht-chain/courbes.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # écrit un fichier image, sans fenêtre
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
# Palette de référence (thème clair) : emplacements 1 et 2 pour les deux courbes, encres pour le texte.
SERIES = {"train_loss": ("entraînement", "#2a78d6"), "val_loss": ("validation", "#eb6834")}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def plot(runs, out):
    histories = {}
    for run in runs:
        path = REPO_DIR / "logs" / f"history_{run}.json"
        if path.exists():
            histories[run] = json.loads(path.read_text())
    if not histories:
        raise SystemExit("Aucun historique trouvé dans logs/history_<run>.json")
    cols = min(3, len(histories))
    rows = math.ceil(len(histories) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows + 0.9), squeeze=False, facecolor=SURFACE)
    for ax in axes.flat[len(histories):]:
        ax.set_visible(False)
    for ax, (run, history) in zip(axes.flat, histories.items()):
        ax.set_facecolor(SURFACE)
        epochs = [h["epoch"] for h in history]
        finals = []
        for key, (label, color) in SERIES.items():
            values = [h[key] for h in history]
            ax.plot(epochs, values, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
            ax.plot(epochs[-1:], values[-1:], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            finals.append(f"{label} {values[-1]:.3f}")
        # Valeurs finales dans le sous-titre, à l'encre : pas d'étiquettes qui se chevauchent au bout des courbes.
        ax.set_title(f"{run}\nfin (epoch {epochs[-1]}) : " + " · ".join(finals), fontsize=8.5, color=INK, loc="left")
        if len(epochs) == 1:
            ax.set_xticks(epochs)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
        ax.margins(x=0.05, y=0.15)
    handles = [plt.Line2D([], [], color=c, linewidth=2) for _, c in SERIES.values()]
    fig.legend(handles, [label for label, _ in SERIES.values()], loc="upper right", frameon=False,
               fontsize=8.5, labelcolor=INK_2, ncol=2)
    fig.suptitle("Perte à chaque epoch (échelle propre à chaque modèle)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Le taux d'apprentissage descend jusqu'à 0 à la fin : un aplatissement final n'est pas "
             "forcément un vrai plateau.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Graphique écrit : {out}")


def main():
    parser = argparse.ArgumentParser(description="Courbes de perte par epoch.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.runs, args.out)


if __name__ == "__main__":
    main()
