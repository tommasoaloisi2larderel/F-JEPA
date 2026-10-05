# Résultats pusht (taux de succès en %)

| modèle | description | epoch | épisodes | itérations CEM | propre | bruité |
|---|---|---|---|---|---|---|
| lewm-officiel | LeWM officiel (référence) |  | 50 | 10 | 94.0 | 60.0 |
| lewm-officiel | LeWM officiel (référence) |  | 50 | 30 | 94.0 | 58.0 |
| lewm-officiel | LeWM officiel (référence) |  | 200 | 10 | 80.5 | 55.5 |
| lewm-pusht-s0 | LeWM réentraîné (étape 0) | 10 | 50 | 10 | 98.0 | 60.0 |
| lewm-pusht-s0 | LeWM réentraîné (étape 0) | 20 | 50 | 10 | 94.0 | 62.0 |
| lewm-pusht-s0 | LeWM réentraîné (étape 0) | 20 | 50 | 30 | 96.0 | 64.0 |
| lewm-pusht-s0 | LeWM réentraîné (étape 0) | 20 | 200 | 10 | 86.5 | 54.5 |
| ar_det-pusht-s0 | D : leur réseau, pas à pas, sans bruit (étape 1) | 10 | 50 | 10 | 88.0 | 56.0 |
| ar_det-pusht-s0 | D : leur réseau, pas à pas, sans bruit (étape 1) | 20 | 50 | 10 | 92.0 | 54.0 |
| ar_det-pusht-s0 | D : leur réseau, pas à pas, sans bruit (étape 1) | 20 | 50 | 30 | 92.0 | 58.0 |
| ar_det-pusht-s0 | D : leur réseau, pas à pas, sans bruit (étape 1) | 20 | 200 | 10 | 86.5 | 58.0 |
| ar_flow-pusht-s0 | C : + flow matching (étape 2a) | 10 | 50 | 10 | 90.0 | 50.0 |
| ar_flow-pusht-s0 | C : + flow matching (étape 2a) | 20 | 50 | 10 | 86.0 | 54.0 |
| ar_flow-pusht-s0 | C : + flow matching (étape 2a) | 20 | 50 | 30 | 90.0 | 56.0 |
| ar_flow-pusht-s0 | C : + flow matching (étape 2a) | 20 | 200 | 10 | 80.0 | 48.0 |
| joint_det-pusht-s0 | B : + prédiction d'un coup (étape 2b) | 10 | 50 | 10 | 90.0 | 58.0 |
| joint_det-pusht-s0 | B : + prédiction d'un coup (étape 2b) | 20 | 50 | 10 | 100.0 | 62.0 |
| joint_det-pusht-s0 | B : + prédiction d'un coup (étape 2b) | 20 | 50 | 30 | 96.0 | 62.0 |
| joint_det-pusht-s0 | B : + prédiction d'un coup (étape 2b) | 20 | 200 | 10 | 87.5 | 50.5 |
| joint_flow-pusht-s0 | A : Flow-JEPA de l'article (étape 3) | 10 | 50 | 10 | 78.0 | 46.0 |
| joint_flow-pusht-s0 | A : Flow-JEPA de l'article (étape 3) | 20 | 50 | 10 | 86.0 | 48.0 |
| joint_flow-pusht-s0 | A : Flow-JEPA de l'article (étape 3) | 20 | 50 | 30 | 92.0 | 62.0 |
| joint_flow-pusht-s0 | A : Flow-JEPA de l'article (étape 3) | 20 | 200 | 10 | 83.5 | 44.5 |
| joint_flow-pusht-s1 | A : Flow-JEPA de l'article (étape 3) | 10 | 50 | 10 | 82.0 | 48.0 |
| joint_flow-pusht-s1 | A : Flow-JEPA de l'article (étape 3) | 20 | 50 | 10 | 86.0 | 44.0 |
| joint_flow-pusht-s1 | A : Flow-JEPA de l'article (étape 3) | 20 | 50 | 30 | 92.0 | 56.0 |
| joint_flow-pusht-s1 | A : Flow-JEPA de l'article (étape 3) | 20 | 200 | 10 | 86.0 | 44.0 |
| random | actions au hasard |  | 50 | 10 | 0.0 | — |
| random | actions au hasard |  | 200 | 10 | 5.0 | — |
