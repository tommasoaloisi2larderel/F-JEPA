"""Lance eval.py de Flow-JEPA sur le checkpoint LeWM (la baseline).

eval.py appelle model.set_flow_seed(), qui n'existe que dans Flow-JEPA.
LeWM est déterministe : on lui ajoute une méthode qui ne fait rien, puis on lance eval.py tel quel.
Les arguments de la ligne de commande sont transmis à eval.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "Flow-JEPA"))

from stable_worldmodel.wm.lewm import LeWM  # noqa: E402

LeWM.set_flow_seed = lambda self, seed: self

import eval as fjepa_eval  # noqa: E402  (eval.py du repo Flow-JEPA)

fjepa_eval.run()
