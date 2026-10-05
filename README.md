# F-JEPA Workspace

This folder keeps our Flow-JEPA work organized around the upstream implementation:

- `Flow-JEPA/`: vendored source from <https://github.com/HuoYanchen/Flow-JEPA>
- `Papers/`: local papers and notes
- `stable-wm/`: local datasets, checkpoints, and run artifacts
- `.uv-cache/`: project-local `uv` cache
- `.uv-python/`: project-local Python versions installed by `uv`
- `.cache/`: project-local runtime caches for Matplotlib, fontconfig, and Hugging Face

The vendored `Flow-JEPA/` source started from upstream commit
`ab73e7c50435e3a1fffb4d3f6d175445e57a2f4e`. It is tracked directly in this
repository so we can modify it freely for our experiments.

## Environment

From the upstream repository folder:

```bash
cd /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA/Flow-JEPA
source /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA/env.sh
uv venv --python=3.10
source .venv/bin/activate
uv pip install "stable-worldmodel[train,env,format]"
uv pip install "datasets>=3,<5" "pyarrow>=20,<25"
```

The explicit `datasets` and `pyarrow` pins keep `stable_pretraining` compatible
with the current package resolver.

## Storage

Use one explicit storage root for datasets and checkpoints:

```bash
source /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA/env.sh
mkdir -p "$STABLEWM_HOME" "$STABLEWM_HOME/checkpoints"
```

Flow-JEPA training passes `$LOCAL_DATASET_DIR` directly to the loader. The
download helper stores the `.h5` files under `$STABLEWM_HOME/datasets`:

| Task | Training config | Dataset file |
| --- | --- | --- |
| Two-Room | `data=tworoom` | `datasets/tworoom.h5` |
| Reacher | `data=dmc` | `datasets/reacher.h5` |
| Push-T | `data=pusht` | `datasets/pusht_expert_train.h5` |
| OGBench-Cube | `data=ogb` | `datasets/cube_single_expert.h5` |

See `DATASETS.md` for the current installed/missing dataset status.

Download helpers:

```bash
cd /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA
./download_dataset.sh tworoom
./download_dataset.sh pusht
./download_dataset.sh reacher
./download_dataset.sh cube
```

`./download_dataset.sh all` downloads all four datasets. The datasets are large,
so check free space first with `df -h .`; the script streams archives directly
into `stable-wm` to avoid duplicate archive copies.

## First Smoke Test

After installing dependencies, verify the environment:

```bash
source /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA/env.sh
cd /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA/Flow-JEPA
source .venv/bin/activate
python -c "import stable_worldmodel as swm; print('stable_worldmodel import OK')"
python train.py --help
```

## Example Training Command

Once the matching dataset is present under `$STABLEWM_HOME/`:

```bash
python train.py data=pusht output_model_name=flow-jepa-pusht
```

## Transition Flow Rollout-2

The current clean Flow-JEPA experiment is:

```text
L_flow_local + lambda * L_rollout_2
```

It trains one shared local flow `z_t -> z_{t+1}` and adds a differentiable
two-step endpoint loss through the composed transition
`G(G(z_t, a_t), a_{t+1})`.

Local or Runpod:

```bash
cd /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA
source ./env.sh
./download_dataset.sh tworoom
cd Flow-JEPA
source .venv/bin/activate
python train.py --config-name tworoom_transition_rollout2
```

Google Colab, from the repository root after cloning or mounting it:

```bash
chmod +x scripts/colab/run_tworoom_transition_rollout2_colab.sh
./scripts/colab/run_tworoom_transition_rollout2_colab.sh
```

Hydra overrides can be appended to the Colab command, for example:

```bash
./scripts/colab/run_tworoom_transition_rollout2_colab.sh trainer.max_epochs=5 loader.batch_size=64
```

Runpod helper scripts now live under `scripts/runpod/`; the Colab entrypoint
lives under `scripts/colab/`.
