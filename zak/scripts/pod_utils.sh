# Fonctions communes aux scripts qui tournent seuls sur le pod (run_full.sh, run_pusht.sh, run_tworoom_variants.sh).
# Chargé avec `source`, après env.sh.

# Si GitHub refuse le token, git échoue tout de suite au lieu d'attendre un mot de passe que personne ne tapera.
export GIT_TERMINAL_PROMPT=0

# Auteur des commits faits par le pod (GIT_NAME et GIT_EMAIL dans env.sh).
git_identity() {
  git -C "$REPO_DIR" config user.name "$GIT_NAME"
  git -C "$REPO_DIR" config user.email "$GIT_EMAIL"
}

# Commit + push d'un dossier du repo. $1 = dossier (relatif au repo), $2 = message de commit.
git_publish() {
  git -C "$REPO_DIR" add "$1"
  git -C "$REPO_DIR" commit -q -m "$2" || true  # rien de nouveau à commiter : pas grave
  git -C "$REPO_DIR" pull -q --rebase --autostash && git -C "$REPO_DIR" push -q
}

push_error_help() {
  echo "ERREUR : impossible de pousser sur GitHub (détail juste au-dessus)."
  echo "Cause la plus probable : le token n'a pas le droit Contents : Read and write."
  echo "Rien n'a été lancé, le pod reste allumé."
}

stop_pod() {
  echo "Arrêt du pod (le disque /workspace est conservé)."
  runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
}

delete_pod() {
  echo "Tout est sur GitHub : suppression du pod."
  runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || stop_pod
}
