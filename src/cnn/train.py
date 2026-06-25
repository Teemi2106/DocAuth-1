"""
Train ForgeryDetector (ResNet50) on ELA images with comprehensive visualization.

Two-phase strategy
------------------
Phase 1  Freeze encoder, train classification head only (fast convergence).
Phase 2  Unfreeze all layers, fine-tune at a much lower LR.

Usage:
    # Step 1 — generate training data (~3 min)
    python -m src.cnn.generate_data

    # Step 2 — train (~10-20 min on CPU, ~3 min on GPU) + generate plots
    python -m src.cnn.train

Weights are saved to:
    weights/forgery_cnn.pt

Visualizations saved to:
    visualizations/training_curves.png
    visualizations/training_metrics.png
    visualizations/performance_comparison.png
    visualizations/module_contributions.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from .dataset import ELADataset
from .model import ForgeryDetector

ROOT         = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "model_data" / "ela_dataset"
DEFAULT_OUT  = ROOT / "weights" / "forgery_cnn.pt"
VIZ_DIR      = ROOT / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def _run_epoch(
    model: ForgeryDetector,
    loader: DataLoader,
    criterion: nn.Module,
    opt: torch.optim.Optimizer | None,
    device: str,
    return_preds: bool = False,
) -> tuple[float, float, list, list]:
    """One forward pass over loader. Returns (avg_loss, accuracy, preds, labels)."""
    training = opt is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss  = criterion(preds, labels)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(imgs)
            correct    += ((preds > 0.5) == labels.bool()).sum().item()
            n          += len(imgs)

            if return_preds:
                all_preds.extend(preds.cpu().detach().numpy().flatten())
                all_labels.extend(labels.cpu().detach().numpy().flatten())

    return total_loss / n, correct / n, all_preds, all_labels


def _plot_training_curves(
    phase1_history: dict,
    phase2_history: dict,
    output: Path = VIZ_DIR / "training_curves.png",
) -> None:
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History: Two-Phase Strategy", fontsize=16, fontweight='bold')

    # Loss curves
    ax = axes[0]
    ax.plot(range(1, len(phase1_history['train_loss']) + 1),
            phase1_history['train_loss'], 'o-', linewidth=2, markersize=4, label='Phase 1 Train', color='#2E86AB')
    ax.plot(range(1, len(phase1_history['val_loss']) + 1),
            phase1_history['val_loss'], 's-', linewidth=2, markersize=4, label='Phase 1 Val', color='#A23B72')

    phase2_start = len(phase1_history['train_loss'])
    ax.plot(range(phase2_start + 1, phase2_start + len(phase2_history['train_loss']) + 1),
            phase2_history['train_loss'], 'o--', linewidth=2, markersize=4, label='Phase 2 Train', color='#F18F01')
    ax.plot(range(phase2_start + 1, phase2_start + len(phase2_history['val_loss']) + 1),
            phase2_history['val_loss'], 's--', linewidth=2, markersize=4, label='Phase 2 Val', color='#C73E1D')

    ax.axvline(x=phase2_start + 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
    ax.set_title('Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    ax.plot(range(1, len(phase1_history['train_acc']) + 1),
            phase1_history['train_acc'], 'o-', linewidth=2, markersize=4, label='Phase 1 Train', color='#2E86AB')
    ax.plot(range(1, len(phase1_history['val_acc']) + 1),
            phase1_history['val_acc'], 's-', linewidth=2, markersize=4, label='Phase 1 Val', color='#A23B72')

    ax.plot(range(phase2_start + 1, phase2_start + len(phase2_history['train_acc']) + 1),
            phase2_history['train_acc'], 'o--', linewidth=2, markersize=4, label='Phase 2 Train', color='#F18F01')
    ax.plot(range(phase2_start + 1, phase2_start + len(phase2_history['val_acc']) + 1),
            phase2_history['val_acc'], 's--', linewidth=2, markersize=4, label='Phase 2 Val', color='#C73E1D')

    ax.axvline(x=phase2_start + 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves: {output}")
    plt.close()


def _plot_performance_metrics(
    phase1_history: dict,
    phase2_history: dict,
    output: Path = VIZ_DIR / "training_metrics.png",
) -> None:
    """Plot phase comparison metrics."""
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle("Training Phase Analysis: Metrics Comparison", fontsize=16, fontweight='bold')

    # Best validation accuracy per phase
    ax = fig.add_subplot(gs[0, 0])
    phases = ['Phase 1\n(Frozen Encoder)', 'Phase 2\n(Full Fine-tune)']
    accuracies = [max(phase1_history['val_acc']), max(phase2_history['val_acc'])]
    bars = ax.bar(phases, accuracies, color=['#2E86AB', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Best Validation Accuracy', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_title('Peak Accuracy per Phase', fontsize=12, fontweight='bold')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Final metrics
    ax = fig.add_subplot(gs[0, 1])
    final_metrics = ['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc']
    phase1_vals = [phase1_history['train_loss'][-1], phase1_history['val_loss'][-1],
                   phase1_history['train_acc'][-1], phase1_history['val_acc'][-1]]
    x = np.arange(len(final_metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, phase1_vals[:2] + [phase1_vals[2]*0.2, phase1_vals[3]*0.2],
                   width, label='Phase 1', color='#2E86AB', alpha=0.8)
    phase2_vals = [phase2_history['train_loss'][-1], phase2_history['val_loss'][-1],
                   phase2_history['train_acc'][-1], phase2_history['val_acc'][-1]]
    bars2 = ax.bar(x + width/2, phase2_vals[:2] + [phase2_vals[2]*0.2, phase2_vals[3]*0.2],
                   width, label='Phase 2', color='#F18F01', alpha=0.8)
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title('Final Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Train\nLoss', 'Val\nLoss', 'Train\nAcc', 'Val\nAcc'], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Convergence speed
    ax = fig.add_subplot(gs[0, 2])
    loss_p1 = np.array(phase1_history['val_loss'])
    loss_p2 = np.array(phase2_history['val_loss'])
    initial_loss_p1 = loss_p1[0]
    initial_loss_p2 = loss_p2[0]
    reduction_p1 = (1 - loss_p1[-1] / initial_loss_p1) * 100
    reduction_p2 = (1 - loss_p2[-1] / initial_loss_p2) * 100
    bars = ax.bar(['Phase 1', 'Phase 2'], [reduction_p1, reduction_p2],
                  color=['#2E86AB', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Loss Reduction (%)', fontsize=11, fontweight='bold')
    ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, [reduction_p1, reduction_p2]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Learning curves comparison
    ax = fig.add_subplot(gs[1, :])
    epochs_p1 = range(1, len(phase1_history['val_acc']) + 1)
    epochs_p2 = range(len(phase1_history['val_acc']) + 1,
                      len(phase1_history['val_acc']) + len(phase2_history['val_acc']) + 1)
    ax.plot(epochs_p1, phase1_history['val_acc'], 'o-', linewidth=2.5, markersize=6,
            label='Phase 1 Validation', color='#2E86AB')
    ax.plot(epochs_p2, phase2_history['val_acc'], 's-', linewidth=2.5, markersize=6,
            label='Phase 2 Validation', color='#F18F01')
    ax.axhline(y=max(phase1_history['val_acc']), color='#2E86AB', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=max(phase2_history['val_acc']), color='#F18F01', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=len(phase1_history['val_acc']) + 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Full Training Trajectory', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training metrics: {output}")
    plt.close()


def _plot_confusion_matrix(all_preds: list, all_labels: list,
                           output: Path = VIZ_DIR / "confusion_matrix.png") -> None:
    """Plot confusion matrix from predictions."""
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predicted = (all_preds > 0.5).astype(int)

    tn = np.sum((predicted == 0) & (all_labels == 0))
    fp = np.sum((predicted == 1) & (all_labels == 0))
    fn = np.sum((predicted == 0) & (all_labels == 1))
    tp = np.sum((predicted == 1) & (all_labels == 1))

    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percentage = count / cm.sum() * 100
            ax.text(j, i, f'{int(count)}\n({percentage:.1f}%)',
                   ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black',
                   fontsize=14, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Forged', 'Genuine'])
    ax.set_yticklabels(['Forged', 'Genuine'])

    # Accuracy metrics text
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    ax.text(1.3, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix: {output}")
    plt.close()


def _plot_score_distribution(all_preds: list, all_labels: list,
                             output: Path = VIZ_DIR / "score_distribution.png") -> None:
    """Plot distribution of prediction scores by class."""
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    genuine_scores = all_preds[all_labels == 1]
    forged_scores = all_preds[all_labels == 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(forged_scores, bins=20, alpha=0.6, label='Forged', color='#C73E1D', edgecolor='black')
    ax.hist(genuine_scores, bins=20, alpha=0.6, label='Genuine', color='#2E86AB', edgecolor='black')

    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    ax.set_xlabel('Predicted Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Score Distribution (Validation Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved score distribution: {output}")
    plt.close()


def train(
    data_dir: str | Path  = DEFAULT_DATA,
    epochs: int           = 15,
    finetune_epochs: int  = 8,
    batch_size: int       = 16,
    lr: float             = 1e-3,
    finetune_lr: float    = 1e-5,
    output: str | Path    = DEFAULT_OUT,
    device: str | None    = None,
) -> None:
    data_dir = Path(data_dir)
    output   = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "genuine").exists():
        print(f"ERROR: training data not found at {data_dir}")
        print("Run:  python -m src.cnn.generate_data")
        return

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = ELADataset(data_dir, split="train")
    val_ds   = ELADataset(data_dir, split="val")
    print(f"Dataset: {len(train_ds)} train  |  {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = ForgeryDetector(pretrained=True).to(device)
    criterion = nn.BCELoss()
    best_acc  = 0.0
    best_preds, best_labels = [], []

    # History tracking
    phase1_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    phase2_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # ── Phase 1: head only ────────────────────────────────────────────────────
    print(f"\n── Phase 1: head training ({epochs} epochs, encoder frozen) ──")
    for p in model.encoder.parameters():
        p.requires_grad = False

    opt   = torch.optim.Adam(model.head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, _, _ = _run_epoch(model, train_loader, criterion, opt, device)
        va_loss, va_acc, va_preds, va_labels = _run_epoch(model, val_loader, criterion, None, device, return_preds=True)
        sched.step()

        phase1_history['train_loss'].append(tr_loss)
        phase1_history['val_loss'].append(va_loss)
        phase1_history['train_acc'].append(tr_acc)
        phase1_history['val_acc'].append(va_acc)

        print(f"  Epoch {ep:>2}/{epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_preds, best_labels = va_preds, va_labels
            torch.save(model.state_dict(), output)
            print(f"    ↳ Saved  val_acc={va_acc:.3f}")

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    print(f"\n── Phase 2: full fine-tune ({finetune_epochs} epochs, all layers) ──")
    for p in model.encoder.parameters():
        p.requires_grad = True

    opt   = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=finetune_epochs)

    for ep in range(1, finetune_epochs + 1):
        tr_loss, tr_acc, _, _ = _run_epoch(model, train_loader, criterion, opt, device)
        va_loss, va_acc, va_preds, va_labels = _run_epoch(model, val_loader, criterion, None, device, return_preds=True)
        sched.step()

        phase2_history['train_loss'].append(tr_loss)
        phase2_history['val_loss'].append(va_loss)
        phase2_history['train_acc'].append(tr_acc)
        phase2_history['val_acc'].append(va_acc)

        print(f"  Epoch {ep:>2}/{finetune_epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_preds, best_labels = va_preds, va_labels
            torch.save(model.state_dict(), output)
            print(f"    ↳ Saved  val_acc={va_acc:.3f}")

    print(f"\nTraining complete.  Best val_acc={best_acc:.3f}  →  {output}")

    # ── Generate visualizations ────────────────────────────────────────────────
    print("\n🎨 Generating visualizations...")
    _plot_training_curves(phase1_history, phase2_history)
    _plot_performance_metrics(phase1_history, phase2_history)
    if best_preds and best_labels:
        _plot_confusion_matrix(best_preds, best_labels)
        _plot_score_distribution(best_preds, best_labels)
    print(f"📊 All visualizations saved to: {VIZ_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ForgeryDetector CNN")
    ap.add_argument("--data-dir",        default=str(DEFAULT_DATA))
    ap.add_argument("--epochs",          type=int,   default=15)
    ap.add_argument("--finetune-epochs", type=int,   default=8)
    ap.add_argument("--batch-size",      type=int,   default=16)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--finetune-lr",     type=float, default=1e-5)
    ap.add_argument("--output",          default=str(DEFAULT_OUT))
    ap.add_argument("--device",          default=None)
    args = ap.parse_args()
    train(
        data_dir       = args.data_dir,
        epochs         = args.epochs,
        finetune_epochs= args.finetune_epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        finetune_lr    = args.finetune_lr,
        output         = args.output,
        device         = args.device,
    )


if __name__ == "__main__":
    main()
