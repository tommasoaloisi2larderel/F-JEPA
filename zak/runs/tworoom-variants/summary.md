# Résultats tworoom (taux de succès en %)

| modèle | description | epoch | épisodes | itérations CEM | propre | bruité |
|---|---|---|---|---|---|---|
| lewm-officiel | LeWM officiel (référence) |  | 50 | 30 | 86.0 | 84.0 |
| lewm-officiel | LeWM officiel (référence) |  | 200 | 30 | 85.0 | 79.5 |
| ar_det-tworoom-e2e-reacherconfig | Direct pas à pas (réseau Flow-JEPA, sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) | 10 | 50 | 30 | 88.0 | 86.0 |
| ar_det-tworoom-e2e-reacherconfig | Direct pas à pas (réseau Flow-JEPA, sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) | 20 | 50 | 30 | 82.0 | 78.0 |
| ar_det-tworoom-e2e-reacherconfig | Direct pas à pas (réseau Flow-JEPA, sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) | 20 | 200 | 30 | 82.5 | 74.5 |
| ar_flow-tworoom-e2e | Flow pas à pas | 10 | 50 | 30 | 88.0 | 62.0 |
| ar_flow-tworoom-e2e | Flow pas à pas | 20 | 50 | 30 | 92.0 | 70.0 |
| ar_flow-tworoom-e2e | Flow pas à pas | 20 | 200 | 30 | 90.0 | 67.5 |
| ar_flow-tworoom-e2e-reacherconfig | Flow pas à pas ; départ de l'état actuel + attention causale (réglages Reacher) | 10 | 50 | 30 | 94.0 | 58.0 |
| ar_flow-tworoom-e2e-reacherconfig | Flow pas à pas ; départ de l'état actuel + attention causale (réglages Reacher) | 20 | 50 | 30 | 88.0 | 52.0 |
| ar_flow-tworoom-e2e-reacherconfig | Flow pas à pas ; départ de l'état actuel + attention causale (réglages Reacher) | 20 | 200 | 30 | 89.5 | 54.0 |
| joint_det-tworoom-e2e | Direct, 5 pas d'un coup (sans flow matching) | 10 | 50 | 30 | 92.0 | 84.0 |
| joint_det-tworoom-e2e | Direct, 5 pas d'un coup (sans flow matching) | 20 | 50 | 30 | 90.0 | 84.0 |
| joint_det-tworoom-e2e | Direct, 5 pas d'un coup (sans flow matching) | 20 | 200 | 30 | 85.5 | 82.0 |
| joint_det-tworoom-e2e-reacherconfig | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) | 10 | 50 | 30 | 96.0 | 80.0 |
| joint_det-tworoom-e2e-reacherconfig | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) | 20 | 50 | 30 | 96.0 | 86.0 |
| joint_det-tworoom-e2e-reacherconfig | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) | 20 | 200 | 30 | 92.0 | 86.0 |
| joint_det-tworoom-e2e-reacherconfig-s1 | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) ; graine 1 | 10 | 50 | 30 | 98.0 | 48.0 |
| joint_det-tworoom-e2e-reacherconfig-s1 | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) ; graine 1 | 20 | 50 | 30 | 94.0 | 90.0 |
| joint_det-tworoom-e2e-reacherconfig-s1 | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel + attention causale (réglages Reacher) ; graine 1 | 20 | 200 | 30 | 92.0 | 82.0 |
| joint_det-tworoom-e2e-reacherconfig-sanscausal | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel, sans attention causale | 10 | 50 | 30 | 94.0 | 86.0 |
| joint_det-tworoom-e2e-reacherconfig-sanscausal | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel, sans attention causale | 20 | 50 | 30 | 96.0 | 94.0 |
| joint_det-tworoom-e2e-reacherconfig-sanscausal | Direct, 5 pas d'un coup (sans flow matching) ; départ de l'état actuel, sans attention causale | 20 | 200 | 30 | 91.0 | 86.5 |
| fjepa-tworoom-full | Flow-JEPA de l'article (5 pas d'un coup + flow matching) | 20 | 50 | 30 | 98.0 | 62.0 |
| fjepa-tworoom-full | Flow-JEPA de l'article (5 pas d'un coup + flow matching) | 20 | 200 | 30 | 94.0 | 63.5 |
| fjepa-tworoom-full-s1 | Flow-JEPA de l'article (5 pas d'un coup + flow matching) ; graine 1 | 10 | 50 | 30 | 98.0 | 94.0 |
| fjepa-tworoom-full-s1 | Flow-JEPA de l'article (5 pas d'un coup + flow matching) ; graine 1 | 20 | 50 | 30 | 100.0 | 84.0 |
| fjepa-tworoom-full-s1 | Flow-JEPA de l'article (5 pas d'un coup + flow matching) ; graine 1 | 20 | 200 | 30 | 98.0 | 83.5 |
| random | actions au hasard |  | 50 | 30 | 32.0 | — |
| random | actions au hasard |  | 200 | 30 | 24.5 | — |
