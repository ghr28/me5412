"""
End-to-end demo pipeline
1) Generate simple multi-condition gait data
2) Train a tiny conditional GAN briefly and synthesize extra samples
3) Train a Hybrid gait classifier on combined data
4) Evaluate and save artifacts

This is a quick runnable demo (small epochs) to validate the repo.
"""

import os
import sys
import json
import time
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Make src importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.synthetic.generate_data import (
    SyntheticDataGenerator,
    create_sample_training_data,
)
from src.models.train import (
    GaitDataset,
    GaitModelTrainer,
    HybridGaitClassifier,
)


def _ensure_dirs():
    os.makedirs(os.path.join(ROOT, "data", "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)


def _minmax_scale_per_feature(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale to [-1, 1] per feature across the dataset.
    Args:
        x: (N, T, C)
    Returns:
        x_scaled, mins, maxs
    """
    mins = x.min(axis=(0, 1), keepdims=True)
    maxs = x.max(axis=(0, 1), keepdims=True)
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    x01 = (x - mins) / denom
    x_scaled = x01 * 2.0 - 1.0
    return x_scaled.astype(np.float32), mins.squeeze(), maxs.squeeze()


def _train_quick_gan(real_data: np.ndarray) -> SyntheticDataGenerator:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = SyntheticDataGenerator(device=device)
    # Keep epochs tiny for a fast demo run
    gen.train_gan(real_data, model_type="conditional", epochs=5, batch_size=32, lr=2e-4)
    return gen


def _synthesize_by_class(gen: SyntheticDataGenerator, per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    data_list = []
    labels_list = []
    for cond, idx in gen.condition_map.items():
        synth = gen.generate_synthetic_data(num_samples=per_class, condition=cond, model_type="conditional")
        data_list.append(synth)
        labels_list.append(np.full((synth.shape[0],), idx, dtype=np.int64))
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return data.astype(np.float32), labels


def _split(data: np.ndarray, labels: np.ndarray, train=0.7, val=0.15):
    N = data.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    data, labels = data[idx], labels[idx]
    n_train = int(N * train)
    n_val = int(N * val)
    train_data, train_labels = data[:n_train], labels[:n_train]
    val_data, val_labels = data[n_train:n_train + n_val], labels[n_train:n_train + n_val]
    test_data, test_labels = data[n_train + n_val:], labels[n_train + n_val:]
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def main():
    t0 = time.time()
    _ensure_dirs()

    # 1) Base multi-condition real-ish data
    print("[1/5] Creating base multi-condition data…")
    real_data, real_labels = create_sample_training_data()  # (N, 100, 6), labels in {0..3}

    # Normalize for GAN stability
    real_scaled, mins, maxs = _minmax_scale_per_feature(real_data)

    # 2) Train a tiny conditional GAN and synthesize extra data
    print("[2/5] Training tiny conditional GAN (few epochs for demo)…")
    gen = _train_quick_gan(real_scaled)

    print("[3/5] Generating synthetic samples per class…")
    synth_data, synth_labels = _synthesize_by_class(gen, per_class=100)

    # 3) Combine real and synthetic data
    print("[4/5] Building train/val/test splits…")
    all_data = np.concatenate([real_scaled.astype(np.float32), synth_data], axis=0)
    all_labels = np.concatenate([real_labels.astype(np.int64), synth_labels], axis=0)

    (trX, trY), (vaX, vaY), (teX, teY) = _split(all_data, all_labels, train=0.7, val=0.15)

    # 4) Train a quick Hybrid classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridGaitClassifier(input_dim=6, num_classes=4)
    trainer = GaitModelTrainer(model, device=device)

    train_loader = DataLoader(GaitDataset(trX, trY), batch_size=32, shuffle=True)
    val_loader = DataLoader(GaitDataset(vaX, vaY), batch_size=32, shuffle=False)
    test_loader = DataLoader(GaitDataset(teX, teY), batch_size=32, shuffle=False)

    print("[5/5] Training classifier (few epochs for demo)…")
    history = trainer.train(train_loader, val_loader, epochs=8, lr=1e-3, weight_decay=1e-4, patience=4)
    metrics = trainer.evaluate(test_loader)

    # Save artifacts
    models_dir = os.path.join(ROOT, "models")
    trainer.save_model(os.path.join(models_dir, "hybrid_demo_model.pth"))

    # Persist results
    results = {"history": history, "metrics": metrics}
    with open(os.path.join(models_dir, "demo_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Persist a small batch of synthetic data for inspection
    data_dir = os.path.join(ROOT, "data", "synthetic")
    np.save(os.path.join(data_dir, "demo_synth_data.npy"), synth_data[:200])
    np.save(os.path.join(data_dir, "demo_synth_labels.npy"), synth_labels[:200])

    dt = time.time() - t0
    print("\n=== Demo pipeline completed ===")
    print(f"Time elapsed: {dt:.1f}s")
    print(f"Saved model: {os.path.join(models_dir, 'hybrid_demo_model.pth')}")
    print(f"Results JSON: {os.path.join(models_dir, 'demo_results.json')}")
    print(f"Sample synthetic data: {os.path.join(data_dir, 'demo_synth_data.npy')}")


if __name__ == "__main__":
    # Reproducibility (best-effort)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
