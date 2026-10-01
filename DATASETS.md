# Dataset Status

Datasets live at:

```bash
/Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA/stable-wm
```

| Task | Config | Expected file | Status |
| --- | --- | --- | --- |
| Two-Room | `data=tworoom` | `tworoom.h5` | Installed |
| Push-T | `data=pusht` | `pusht_expert_train.h5` | Not installed |
| Reacher | `data=dmc` | `reacher.h5` | Not installed |
| OGBench-Cube | `data=ogb` | `cube_single_expert.h5` | Not installed |

## Commands

```bash
cd /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA
./download_dataset.sh tworoom
./download_dataset.sh pusht
./download_dataset.sh reacher
./download_dataset.sh cube
```

Check installed datasets:

```bash
cd /Users/tommasoaloisi/ENPC/Stage/MGB/Mini-projects/F-JEPA/Flow-JEPA
source ../env.sh
.venv/bin/swm datasets
.venv/bin/swm inspect tworoom
```

## Disk Note

As of setup, Two-Room occupies about 12 GB after extraction. The remaining
datasets are large enough that they should be downloaded after freeing disk
space or moving `STABLEWM_HOME` to a larger volume.
