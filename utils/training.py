"""Shared training and evaluation helpers for classification notebooks."""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler=None,
        criterion=None,
        device: str = "cuda",
        mixed_precision: bool = True,
        grad_accum_steps: int = 1,
        early_stopping_patience: int = 5,
        checkpoint_path: str = "best_model.pth",
        experiment_name: str = "experiment",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.mixed_precision = mixed_precision
        self.grad_accum_steps = grad_accum_steps
        self.patience = early_stopping_patience
        self.checkpoint_path = checkpoint_path
        self.experiment_name = experiment_name

        self.scaler = GradScaler() if mixed_precision else None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, train_loader) -> float:
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        self.optimizer.zero_grad()

        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.mixed_precision:
                scaler = self.scaler
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) / self.grad_accum_steps
                if scaler is None:
                    raise RuntimeError(
                        "GradScaler is required when mixed_precision is enabled."
                    )
                scaler.scale(loss).backward()

                if (batch_index + 1) % self.grad_accum_steps == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / self.grad_accum_steps
                loss.backward()

                if (batch_index + 1) % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item() * self.grad_accum_steps
            batch_count += 1

        return running_loss / batch_count

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float | np.ndarray]:
        self.model.eval()
        all_predictions = []
        all_labels = []
        running_loss = 0.0
        batch_count = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            predictions = outputs.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()
            batch_count += 1

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        return {
            "loss": running_loss / batch_count,
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1_macro": f1_score(
                all_labels, all_predictions, average="macro", zero_division=0
            ),
            "f1_weighted": f1_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            ),
            "preds": all_predictions,
            "labels": all_labels,
        }

    def fit(self, train_loader, val_loader, epochs: int) -> dict[str, list[float]]:
        print(f"Training {self.experiment_name}")
        print(f"Device: {self.device} | Epochs: {epochs} | AMP: {self.mixed_precision}")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            elapsed_seconds = time.time() - start_time

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_macro"])

            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1_macro']:.4f} | "
                f"time={elapsed_seconds:.1f}s"
            )

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_metrics["f1_macro"])
                except TypeError:
                    self.scheduler.step()

            if val_metrics["f1_macro"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_macro"]
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Saved checkpoint to {self.checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Stopped early after epoch {epoch}")
                    break

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        return self.history


def full_evaluation(
    model: nn.Module,
    test_loader,
    device: str,
    class_names: list[str],
    experiment_name: str,
    results_dir: str,
    mixed_precision: bool = True,
) -> dict[str, float | dict]:
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            if mixed_precision:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            predictions = outputs.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    print(f"Final test results for {experiment_name}")
    print(
        f"accuracy={accuracy:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weighted:.4f}"
    )

    report_text = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0,
    )
    print(report_text)

    report_dict = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"{experiment_name}_report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report_dict, file, indent=2)

    confusion = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(confusion, class_names, experiment_name, results_dir)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report_dict,
    }


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    experiment_name: str,
    results_dir: str,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Confusion Matrix - {experiment_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(results_dir, f"{experiment_name}_confusion_matrix.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved confusion matrix to {output_path}")


def plot_training_history(
    history: dict[str, list[float]], experiment_name: str, results_dir: str
) -> None:
    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    loss_axis.plot(epochs, history["train_loss"], label="Train Loss")
    loss_axis.plot(epochs, history["val_loss"], label="Val Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Loss Curves")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)

    metric_axis.plot(epochs, history["val_acc"], label="Val Accuracy")
    metric_axis.plot(epochs, history["val_f1"], label="Val F1")
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel("Score")
    metric_axis.set_title("Validation Metrics")
    metric_axis.legend()
    metric_axis.grid(True, alpha=0.3)

    figure.suptitle(experiment_name)
    plt.tight_layout()

    output_path = os.path.join(results_dir, f"{experiment_name}_training_curves.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved training curves to {output_path}")


def plot_comparison_bar(results: dict[str, dict[str, float]], results_dir: str) -> None:
    names = list(results.keys())
    accuracy_values = [results[name]["accuracy"] for name in names]
    f1_values = [results[name]["f1_macro"] for name in names]

    x_axis = np.arange(len(names))
    width = 0.35

    figure, axis = plt.subplots(figsize=(10, 6))
    accuracy_bars = axis.bar(
        x_axis - width / 2, accuracy_values, width, label="Accuracy"
    )
    f1_bars = axis.bar(x_axis + width / 2, f1_values, width, label="F1 (macro)")

    axis.set_ylabel("Score")
    axis.set_title("Model Comparison - iWildCam 2019")
    axis.set_xticks(x_axis)
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.legend()
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.3)

    for bar in accuracy_bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in f1_bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = os.path.join(results_dir, "model_comparison.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved comparison chart to {output_path}")
