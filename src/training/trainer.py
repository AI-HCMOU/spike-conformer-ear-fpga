"""
Training loop with AMP, gradient clipping, checkpointing, and TensorBoard logging.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from ..evaluation.metrics import MetricTracker
from ..models.losses import ArcFaceLoss


class Trainer:
    """Handles full training lifecycle."""

    def __init__(self, model: nn.Module, optimizer, scheduler, criterion,
                 device: torch.device, cfg: dict,
                 writer: Optional[SummaryWriter] = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.writer = writer or SummaryWriter("runs/spikeconformer")

        train_cfg = cfg["training"]
        self.use_amp = train_cfg.get("mixed_precision", True)
        self.grad_clip = train_cfg.get("gradient_clip", 1.0)
        self.epochs = train_cfg["epochs"]
        self.patience = train_cfg.get("patience", 20)

        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_metric = 0.0
        self.save_dir = Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True)

        # Metrics tracker
        eval_metrics = cfg["evaluation"]["metrics"]
        self.metrics = MetricTracker(eval_metrics)

        # Whether using ArcFace (requires embeddings, not logits)
        self.use_arcface = isinstance(criterion, ArcFaceLoss)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0):
        """Main training loop."""
        no_improve = 0

        for epoch in range(start_epoch, self.epochs):
            train_loss = self._train_epoch(train_loader, epoch)
            val_results = self.evaluate(val_loader)

            # Step scheduler
            self.scheduler.step()

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("LR", lr, epoch)
            for k, v in val_results.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            # Checkpoint on primary metric
            primary = val_results.get("rank1", val_results.get("accuracy", 0))
            if primary > self.best_metric:
                self.best_metric = primary
                self._save_checkpoint(epoch, is_best=True)
                no_improve = 0
            else:
                no_improve += 1

            # Print epoch summary
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_results.items())
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {train_loss:.4f} | "
                  f"{metrics_str} | LR: {lr:.6f} | Best: {self.best_metric:.4f}")

            # Early stopping
            if no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # Save latest
            self._save_checkpoint(epoch, is_best=False)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Train [{epoch+1}]", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                if self.use_arcface:
                    embeddings = self.model.get_embeddings(inputs)
                    loss = self.criterion(embeddings, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate model on dataloader."""
        self.model.eval()
        self.metrics.reset()

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=False):
                outputs = self.model(inputs)

            self.metrics.update(outputs, targets)

        return self.metrics.compute()

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.cfg,
        }
        torch.save(state, self.save_dir / "last.pth")
        if is_best:
            torch.save(state, self.save_dir / "best.pth")

    def resume(self, path: str) -> int:
        """Resume training from checkpoint. Returns next epoch."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_metric = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']+1}, best metric: {self.best_metric:.4f}")
        return ckpt["epoch"] + 1
