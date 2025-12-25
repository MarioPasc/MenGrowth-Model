"""System-level metrics callbacks for monitoring training efficiency.

This module provides callbacks for tracking hardware utilization and
training throughput, useful for optimization and resource monitoring.
"""

import time
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from typing import Any


class SystemMetricsCallback(Callback):
    """Logs GPU memory usage and training throughput.

    Tracks:
    - GPU memory allocated/reserved (in GB)
    - Samples processed per second (epoch-level)
    - Batch processing time (in milliseconds)

    Metrics are logged with minimal overhead (<0.1% training time) by
    sampling every 50 batches for batch-level metrics.

    Example:
        >>> callback = SystemMetricsCallback()
        >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(self):
        super().__init__()
        self.batch_start_time = None
        self.epoch_start_time = None
        self.epoch_samples_processed = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset epoch-level counters.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LightningModule being trained
        """
        self.epoch_start_time = time.time()
        self.epoch_samples_processed = 0

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Record batch start time.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LightningModule being trained
            batch: Current batch
            batch_idx: Batch index
        """
        self.batch_start_time = time.time()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Log system metrics after each batch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LightningModule being trained
            outputs: Training step outputs
            batch: Current batch
            batch_idx: Batch index
        """
        # Batch processing time
        if self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
            batch_size = batch["image"].size(0)
            self.epoch_samples_processed += batch_size

            # Log every 50 steps to avoid overhead
            if batch_idx % 50 == 0:
                pl_module.log(
                    "system/batch_time_ms",
                    batch_time * 1000,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

        # GPU memory (only if CUDA available)
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

            pl_module.log("system/gpu_mem_allocated_gb", gpu_mem_allocated, on_step=True, on_epoch=False)
            pl_module.log("system/gpu_mem_reserved_gb", gpu_mem_reserved, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log epoch-level throughput.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LightningModule being trained
        """
        if self.epoch_start_time is not None and self.epoch_samples_processed > 0:
            epoch_time = time.time() - self.epoch_start_time
            throughput = self.epoch_samples_processed / epoch_time

            pl_module.log(
                "system/samples_per_sec",
                throughput,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
