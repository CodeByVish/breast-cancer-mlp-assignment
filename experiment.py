"""Development-only cross-validation and held-out testing for UCI WDBC."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib-cache").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_PATH = Path("data/wdbc.data")
RESULTS_DIR = Path("results")
BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension",
]
FEATURE_NAMES = (
    [f"mean_{name}" for name in BASE_FEATURES]
    + [f"se_{name}" for name in BASE_FEATURES]
    + [f"worst_{name}" for name in BASE_FEATURES]
)
ARCHITECTURES = {
    "MLP-1 (16)": (16,),
    "MLP-2 (16-8)": (16, 8),
    "MLP-3 (32-16)": (32, 16),
    "MLP-4 (32-16-8)": (32, 16, 8),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["id", "diagnosis", *FEATURE_NAMES]
    if path.exists():
        raw = pd.read_csv(path, header=None, names=columns)
        X = raw[FEATURE_NAMES].copy()
        y = raw["diagnosis"].map({"B": 0, "M": 1}).to_numpy(dtype=int)
    else:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features.copy()
        X.columns = FEATURE_NAMES
        target = dataset.data.targets.iloc[:, 0].astype(str)
        y = target.map({"B": 0, "M": 1}).to_numpy(dtype=int)

    assert X.shape == (569, 30), f"Unexpected feature shape: {X.shape}"
    assert int(X.isna().sum().sum()) == 0, "Features contain missing values"
    assert np.bincount(y).tolist() == [357, 212], "Unexpected class counts"
    return X, y


def build_mlp(input_dim: int, hidden_layers: tuple[int, ...]) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                units, activation="relu", kernel_regularizer=regularizer
            )
        )
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


def eer_operating_point(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities, pos_label=1)
    fnr = 1.0 - tpr
    finite = np.isfinite(thresholds)
    candidate_indices = np.flatnonzero(finite)
    idx = candidate_indices[np.argmin(np.abs(fpr[finite] - fnr[finite]))]
    return {
        "fpr": fpr,
        "tpr": tpr,
        "threshold": float(thresholds[idx]),
        "eer": float((fpr[idx] + fnr[idx]) / 2.0),
        "index": int(idx),
    }


def evaluate_probabilities(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    """Evaluate a threshold already chosen on development data."""
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, probability)),
        "mean_error_rate": float((fp / (tn + fp) + fn / (tp + fn)) / 2),
        "fpr": float(fp / (tn + fp)),
        "fnr": float(fn / (tp + fn)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_mlp_cv(X: pd.DataFrame, y: np.ndarray, hidden_layers: tuple[int, ...]):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    fold_rows = []
    histories = []

    for fold, (outer_train, outer_valid) in enumerate(splitter.split(X, y), start=1):
        fit_idx, stop_idx = train_test_split(
            outer_train,
            test_size=0.15,
            stratify=y[outer_train],
            random_state=SEED + fold,
        )
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X.iloc[fit_idx])
        X_stop = scaler.transform(X.iloc[stop_idx])
        X_valid = scaler.transform(X.iloc[outer_valid])

        tf.keras.backend.clear_session()
        set_seed(SEED + fold)
        model = build_mlp(X.shape[1], hidden_layers)
        history = model.fit(
            X_fit,
            y[fit_idx],
            validation_data=(X_stop, y[stop_idx]),
            epochs=200,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                )
            ],
            verbose=0,
        )
        histories.append(history.history)
        probability = model(X_valid, training=False).numpy().ravel()
        oof[outer_valid] = probability
        point = eer_operating_point(y[outer_valid], probability)
        fold_rows.append(
            {
                "fold": fold,
                "auc": float(roc_auc_score(y[outer_valid], probability)),
                "eer": point["eer"],
                "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                "epochs": len(history.history["loss"]),
                "parameters": model.count_params(),
            }
        )
    return oof, pd.DataFrame(fold_rows), histories


def run_logistic_cv(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in splitter.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_valid = scaler.transform(X.iloc[valid_idx])
        model = LogisticRegression(max_iter=2000, random_state=SEED)
        model.fit(X_train, y[train_idx])
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
    return oof


def save_plots(y: np.ndarray, probability: np.ndarray, metrics: dict) -> None:
    fpr, tpr, _ = roc_curve(y, probability)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"MLP (AUC={metrics['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.scatter(metrics["fpr"], metrics["sensitivity"], color="crimson", zorder=3,
               label=f"Development threshold={metrics['threshold']:.3f}")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Held-out test ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    matrix = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Test confusion matrix at development threshold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    set_seed(SEED)
    X, y = load_data()
    dev_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=SEED
    )
    X_dev, y_dev = X.iloc[dev_idx], y[dev_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    summaries, probabilities, fold_tables, histories = [], {}, {}, {}
    for name, layout in ARCHITECTURES.items():
        print(f"Development cross-validation: {name}", flush=True)
        oof, folds, histories[name] = run_mlp_cv(X_dev, y_dev, layout)
        probabilities[name], fold_tables[name] = oof, folds
        point = eer_operating_point(y_dev, oof)
        summaries.append({
            "model": name, "development_eer": point["eer"],
            "development_auc": float(roc_auc_score(y_dev, oof)),
            "parameters": int(folds.parameters.iloc[0]),
        })
        folds.to_csv(RESULTS_DIR / f"{name.split()[0].lower()}_fold_metrics.csv", index=False)
    comparison = pd.DataFrame(summaries).sort_values(
        ["development_eer", "development_auc", "parameters"], ascending=[True, False, True]
    )
    comparison.to_csv(RESULTS_DIR / "architecture_comparison.csv", index=False)
    selected = comparison.iloc[0]["model"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    for fold, (ax, losses) in enumerate(zip(axes, histories[selected]), 1):
        ax.plot(np.arange(1, len(losses["loss"]) + 1), losses["loss"], label="Training")
        ax.plot(np.arange(1, len(losses["val_loss"]) + 1), losses["val_loss"], label="Early-stopping validation")
        ax.set(title=f"Development fold {fold}", xlabel="Epoch")
    axes[0].set_ylabel("Loss (cross-entropy + L2)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_curves.png", dpi=160)
    plt.close(fig)
    mlp_threshold = eer_operating_point(y_dev, probabilities[selected])["threshold"]
    # Fix training duration using development validation losses, never test loss.
    epochs = int(np.median(fold_tables[selected]["best_epoch"]))
    logistic_oof = run_logistic_cv(X_dev, y_dev)
    logistic_point = eer_operating_point(y_dev, logistic_oof)
    pd.DataFrame({"row_index": dev_idx, "actual": y_dev,
                  "mlp_probability": probabilities[selected],
                  "logistic_probability": logistic_oof}).to_csv(
                      RESULTS_DIR / "out_of_fold_predictions.csv", index=False)

    scaler = StandardScaler()
    dev_scaled = scaler.fit_transform(X_dev)
    test_scaled = scaler.transform(X_test)
    tf.keras.backend.clear_session()
    set_seed(SEED)
    mlp = build_mlp(30, ARCHITECTURES[selected])
    mlp.fit(dev_scaled, y_dev, epochs=epochs, batch_size=32, verbose=0)
    logistic = LogisticRegression(max_iter=2000, random_state=SEED).fit(dev_scaled, y_dev)
    # All choices above are frozen before either model is evaluated on the test set.
    mlp_probability = mlp(test_scaled, training=False).numpy().ravel()
    logistic_probability = logistic.predict_proba(test_scaled)[:, 1]
    mlp_metrics = evaluate_probabilities(y_test, mlp_probability, mlp_threshold)
    logistic_metrics = evaluate_probabilities(y_test, logistic_probability, logistic_point["threshold"])
    result = {
        "split": {"seed": SEED, "development": len(dev_idx), "test": len(test_idx),
                  "development_class_counts_B_M": np.bincount(y_dev).tolist(),
                  "test_class_counts_B_M": np.bincount(y_test).tolist()},
        "selection": {"model": selected, "final_epochs": epochs,
                      "development_mlp_eer": float(comparison.iloc[0]["development_eer"]),
                      "development_logistic_eer": logistic_point["eer"]},
        "mlp": mlp_metrics, "logistic_regression": logistic_metrics,
    }
    (RESULTS_DIR / "final_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"row_index": test_idx, "actual": y_test,
                  "mlp_probability": mlp_probability,
                  "logistic_probability": logistic_probability}).to_csv(
                      RESULTS_DIR / "test_predictions.csv", index=False)
    save_plots(y_test, mlp_probability, mlp_metrics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
