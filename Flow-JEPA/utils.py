import csv
import time
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer for multiprocessing data loaders."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


class ThroughputProfilerCallback(Callback):
    """Write lightweight per-batch timing for bottleneck checks."""

    def __init__(self, output_path, every_n_steps: int = 10):
        super().__init__()
        self.output_path = Path(output_path)
        self.every_n_steps = max(1, int(every_n_steps))
        self._batch_start = None
        self._last_batch_end = None
        self._writer = None
        self._file = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="")
        fieldnames = [
            "epoch",
            "step",
            "batch_idx",
            "batch_size",
            "data_wait_s",
            "step_s",
            "examples_per_s",
            "gpu_mem_allocated_mb",
            "gpu_mem_reserved_mb",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def on_train_epoch_start(self, trainer, pl_module):
        self._last_batch_end = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.is_global_zero and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self._last_batch_end is None:
            self._data_wait = 0.0
        else:
            self._data_wait = now - self._last_batch_end
        self._batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        step_s = now - self._batch_start if self._batch_start else 0.0
        self._last_batch_end = now

        if (batch_idx + 1) % self.every_n_steps != 0:
            return

        batch_size = self._batch_size(batch)
        row = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "data_wait_s": self._data_wait,
            "step_s": step_s,
            "examples_per_s": batch_size / step_s if step_s > 0 else 0.0,
            "gpu_mem_allocated_mb": self._cuda_mb(torch.cuda.memory_allocated),
            "gpu_mem_reserved_mb": self._cuda_mb(torch.cuda.memory_reserved),
        }
        self._writer.writerow(row)
        self._file.flush()
        print(
            "PROFILE "
            f"step={row['step']} batch={batch_idx} "
            f"data_wait_s={row['data_wait_s']:.4f} "
            f"step_s={row['step_s']:.4f} "
            f"examples_per_s={row['examples_per_s']:.2f} "
            f"gpu_reserved_mb={row['gpu_mem_reserved_mb']:.0f}",
            flush=True,
        )

    def on_fit_end(self, trainer, pl_module):
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def _batch_size(batch):
        if isinstance(batch, dict):
            for value in batch.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        if torch.is_tensor(batch) and batch.ndim > 0:
            return int(batch.shape[0])
        return 0

    @staticmethod
    def _cuda_mb(fn):
        if not torch.cuda.is_available():
            return 0.0
        return float(fn() / 1024**2)
