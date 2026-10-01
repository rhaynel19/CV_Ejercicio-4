from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from inf8239_u02_cv.data import CLASS_NAMES, audit_splits, normalize_images, validate_images
from inf8239_u02_cv.models import build_cnn, build_dense

SEED = 42


def import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except Exception as error:
        raise RuntimeError(
            "TensorFlow no se pudo importar. Use Python 3.12 en Linux, WSL2 o Colab; "
            "no se generan resultados simulados. "
            f"Detalle: {type(error).__name__}: {error}"
        ) from error
    return tf


def save_training_curves(histories: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, history in histories.items():
        axes[0].plot(history["loss"], label=f"{name} entrenamiento")
        axes[0].plot(history["val_loss"], linestyle="--", label=f"{name} validación")
        axes[1].plot(history["val_accuracy"], label=name)
    axes[0].set(title="Pérdida", xlabel="Época", ylabel="Entropía cruzada")
    axes[1].set(title="Exactitud de validación", xlabel="Época", ylabel="Accuracy")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_error_examples(images: np.ndarray, labels: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    errors = np.flatnonzero(predictions != labels)[:20]
    figure, axes = plt.subplots(4, 5, figsize=(11, 9))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, index in zip(axes.ravel(), errors):
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(
            f"Real: {CLASS_NAMES[int(labels[index])] }\nPred.: {CLASS_NAMES[int(predictions[index])]}",
            fontsize=8,
        )
        axis.axis("off")
    if len(errors) == 0:
        axes[0, 0].text(0.5, 0.5, "Sin errores en la muestra", ha="center", va="center")
    figure.suptitle("Errores de la CNN en el conjunto de prueba")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y compara baseline denso y CNN en Fashion-MNIST.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--validation-size", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser al menos 1")
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size debe estar entre 0 y 1")

    tf = import_tensorflow()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    reports_dir = ROOT / "reports"
    models_dir = ROOT / "models"
    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    (images_train_all, labels_train_all), (images_test_raw, labels_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    train_indices, valid_indices = train_test_split(
        np.arange(len(labels_train_all)),
        test_size=args.validation_size,
        random_state=SEED,
        stratify=labels_train_all,
    )
    raw_splits = {
        "train": (images_train_all[train_indices], labels_train_all[train_indices]),
        "validation": (images_train_all[valid_indices], labels_train_all[valid_indices]),
        "test": (images_test_raw, labels_test),
    }
    audit = audit_splits(raw_splits)
    audit.update(
        {
            "seed": SEED,
            "validation_fraction": args.validation_size,
            "class_names": CLASS_NAMES,
            "counts": {
                split_name: split_data["count"]
                for split_name, split_data in audit["splits"].items()
            },
            "split_method": "stratified train_test_split; official test reserved for final evaluation",
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tensorflow": tf.__version__,
                "devices": [device.name for device in tf.config.list_physical_devices()],
            },
        }
    )
    (reports_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    x_train, y_train = raw_splits["train"]
    x_valid, y_valid = raw_splits["validation"]
    x_test, y_test = raw_splits["test"]
    x_train, x_valid, x_test = map(normalize_images, (x_train, x_valid, x_test))
    validate_images(x_train, y_train)
    validate_images(x_valid, y_valid)
    validate_images(x_test, y_test)

    metrics: dict[str, dict] = {}
    histories: dict[str, dict] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    model_factories = {"dense": build_dense, "cnn": build_cnn}

    for name, factory in model_factories.items():
        model = factory(tf)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ]
        if name == "cnn":
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    models_dir / "best_cnn.keras",
                    monitor="val_loss",
                    save_best_only=True,
                )
            )
        started = perf_counter()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=args.epochs,
            batch_size=128,
            callbacks=callbacks,
            verbose=2,
        )
        training_seconds = perf_counter() - started
        model_path = models_dir / f"{name}.keras"
        model.save(model_path)

        model.predict(x_test[:128], verbose=0)
        inference_times = []
        probabilities = None
        for _ in range(3):
            started = perf_counter()
            probabilities = model.predict(x_test, batch_size=128, verbose=0)
            inference_times.append(perf_counter() - started)
        assert probabilities is not None
        predictions = probabilities.argmax(axis=1)
        predictions_by_model[name] = predictions
        report = classification_report(
            y_test,
            predictions,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        metrics[name] = {
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "accuracy": float(np.mean(predictions == y_test)),
            "per_class": report,
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=np.arange(len(CLASS_NAMES)),
            ).tolist(),
            "parameters": int(model.count_params()),
            "model_size_bytes": int(model_path.stat().st_size),
            "train_seconds": float(training_seconds),
            "inference_ms_per_image": float(np.median(inference_times) * 1000 / len(x_test)),
        }
        histories[name] = history.history
        print(
            f"{name}: macro-F1={metrics[name]['f1_macro']:.4f}; "
            f"accuracy={metrics[name]['accuracy']:.4f}; "
            f"parameters={metrics[name]['parameters']:,}; "
            f"size={metrics[name]['model_size_bytes']:,} bytes; "
            f"train={training_seconds:.1f} s; "
            f"inference={metrics[name]['inference_ms_per_image']:.4f} ms/image"
        )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions_by_model["cnn"],
        labels=np.arange(len(CLASS_NAMES)),
        display_labels=CLASS_NAMES,
        cmap="Blues",
        xticks_rotation=45,
    )
    plt.title("CNN: matriz de confusión en test")
    plt.savefig(reports_dir / "confusion_cnn.png", dpi=170, bbox_inches="tight")
    plt.close()
    save_training_curves(histories, reports_dir / "training_curves.png")
    save_error_examples(x_test, y_test, predictions_by_model["cnn"], reports_dir / "cnn_errors.png")
    (reports_dir / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Auditoría, métricas y figuras guardadas en {reports_dir}")


if __name__ == "__main__":
    main()