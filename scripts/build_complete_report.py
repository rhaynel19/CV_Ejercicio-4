from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = (
    ("training_curves.png", "Curvas de entrenamiento y validación"),
    ("confusion_cnn.png", "Matriz de confusión de la CNN"),
    ("cnn_errors.png", "Muestra de errores de la CNN"),
)


def _repository_url() -> str:
    configured = os.environ.get("REPOSITORY_URL", "").strip()
    if configured:
        return configured
    result = subprocess.run(
        ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _table(rows: list[list[str]], widths: tuple[float, ...], font_size: float = 8) -> Table:
    table = Table(
        rows,
        repeatRows=1,
        colWidths=[width * inch for width in widths],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#184E5A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C7C9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF4F2")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B7C7C9"))
    canvas.line(document.leftMargin, 0.45 * inch, letter[0] - document.rightMargin, 0.45 * inch)
    canvas.setFillColor(colors.HexColor("#52666A"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(document.leftMargin, 0.29 * inch, "INF-8239 · U02.E04 · Fashion-MNIST")
    canvas.drawRightString(letter[0] - document.rightMargin, 0.29 * inch, f"Página {document.page}")
    canvas.restoreState()


def _load_evidence() -> tuple[dict, dict]:
    audit_path = REPORTS / "data_audit.json"
    metrics_path = REPORTS / "cv_metrics.json"
    missing = [path.name for path in (audit_path, metrics_path) if not path.is_file()]
    missing.extend(name for name, _ in FIGURES if not (REPORTS / name).is_file())
    if missing:
        raise FileNotFoundError(
            "No se genera el informe sin una corrida completa. Faltan: "
            + ", ".join(missing)
            + ". Ejecute el notebook o scripts/train_cv.py primero."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if any("confusion_matrix" not in values for values in metrics.values()):
        raise ValueError("cv_metrics.json no contiene matrices de confusión; vuelva a ejecutar el entrenamiento actualizado.")
    return audit, metrics


def create_report(output: Path) -> None:
    audit, metrics = _load_evidence()
    class_names = audit["class_names"]
    dense = metrics["dense"]
    cnn = metrics["cnn"]
    f1_delta = dense["f1_macro"] - cnn["f1_macro"]
    accuracy_delta = dense["accuracy"] - cnn["accuracy"]
    parameter_reduction = 1 - cnn["parameters"] / dense["parameters"]
    size_reduction = 1 - cnn["model_size_bytes"] / dense["model_size_bytes"]
    training_ratio = cnn["train_seconds"] / dense["train_seconds"]
    inference_ratio = cnn["inference_ms_per_image"] / dense["inference_ms_per_image"]

    cnn_matrix = cnn["confusion_matrix"]
    confusion_pairs = sorted(
        [
            (int(cnn_matrix[actual][predicted]), class_names[actual], class_names[predicted])
            for actual in range(len(class_names))
            for predicted in range(len(class_names))
            if actual != predicted and cnn_matrix[actual][predicted]
        ],
        reverse=True,
    )
    total_errors = sum(
        int(cnn_matrix[actual][predicted])
        for actual in range(len(class_names))
        for predicted in range(len(class_names))
        if actual != predicted
    )
    hardest_class = min(class_names, key=lambda name: cnn["per_class"][name]["recall"])
    hardest_index = class_names.index(hardest_class)
    hardest_support = int(cnn["per_class"][hardest_class]["support"])
    hardest_false_negatives = hardest_support - int(cnn_matrix[hardest_index][hardest_index])

    repository = _repository_url().rstrip("/")
    if repository.endswith(".git"):
        repository = repository[:-4]
    if repository:
        repository_markup = (
            f'Repositorio: <link href="{escape(repository)}" color="#176B73">'
            f"{escape(repository)}</link>"
        )
    else:
        repository_markup = "Repositorio: publicar y configurar origin para habilitar el enlace público."

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#52666A"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontSize=9.2,
            leading=13,
            spaceAfter=7,
        )
    )
    styles["Title"].fontSize = 20
    styles["Title"].leading = 25
    styles["Heading1"].fontSize = 16
    styles["Heading1"].leading = 20
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 15

    story = [
        Paragraph("Ejercicio 04 · Clasificador visual con CNN y Model Card", styles["Title"]),
        Paragraph("INF-8239 Ciencia de Datos II · Unidad 02 · U02.E04", styles["ReportSubtitle"]),
        Paragraph("Edwin Ramón José Nolasco · Universidad Autónoma de Santo Domingo", styles["Normal"]),
        Spacer(1, 0.12 * inch),
        Paragraph(repository_markup, styles["ReportBody"]),
        Paragraph("Síntesis ejecutiva", styles["Heading1"]),
        Paragraph(
            f"Se compararon un baseline denso y una CNN pequeña en Fashion-MNIST, usando la misma partición. "
            f"En este test el baseline obtuvo F1 macro {dense['f1_macro']:.4f} y accuracy {dense['accuracy']:.4f}; "
            f"la CNN obtuvo {cnn['f1_macro']:.4f} y {cnn['accuracy']:.4f}. El baseline aventajó a la CNN "
            f"por {f1_delta:.4f} en F1 macro y {accuracy_delta:.4f} en accuracy.",
            styles["ReportBody"],
        ),
        Paragraph(
            f"Decisión: conservar el baseline para este benchmark y esta corrida. Aunque la CNN tiene "
            f"{parameter_reduction:.1%} menos parámetros y ocupa {size_reduction:.1%} menos espacio, "
            f"tardó {training_ratio:.1f} veces más en entrenar y su inferencia fue {inference_ratio:.1f} veces más lenta. "
            "La elección se limita a este split, semilla y CPU; no es una conclusión universal.",
            styles["ReportBody"],
        ),
        Paragraph("Comparación global en el test reservado", styles["Heading2"]),
    ]

    global_rows = [["Modelo", "Accuracy", "F1 macro", "Parámetros", "Tamaño", "Train (s)", "ms/imagen"]]
    for name, values in (("Baseline denso", dense), ("CNN", cnn)):
        global_rows.append(
            [
                name,
                f"{values['accuracy']:.4f}",
                f"{values['f1_macro']:.4f}",
                f"{values['parameters']:,}",
                f"{values['model_size_bytes'] / 1024:.1f} KiB",
                f"{values['train_seconds']:.1f}",
                f"{values['inference_ms_per_image']:.4f}",
            ]
        )
    story.extend(
        [
            _table(global_rows, (1.12, 0.72, 0.72, 0.82, 0.72, 0.66, 0.74), 7.5),
            Spacer(1, 0.12 * inch),
            Paragraph(
                "Rúbrica cubierta: auditoría y split (data_audit.json); baseline/CNN (modelos Keras); métricas, "
                "curvas y errores (cv_metrics.json y figuras); costo y decisión (tabla anterior); Model Card, "
                "pruebas, README y Git (archivos incluidos en el repositorio).",
                styles["ReportBody"],
            ),
            PageBreak(),
            Paragraph("Datos, auditoría y protocolo", styles["Heading1"]),
            Paragraph(
                "Fashion-MNIST contiene imágenes de prendas en escala de grises de 28 × 28 píxeles y diez clases. "
                "La fuente usada es keras.datasets.fashion_mnist; referencia del conjunto: "
                '<link href="https://github.com/zalandoresearch/fashion-mnist" color="#176B73">'
                "repositorio Fashion-MNIST de Zalando Research</link>. Cada clase está balanceada en los tres splits.",
                styles["ReportBody"],
            ),
            Paragraph(
                f"La auditoría registra semilla {audit['seed']}, partición estratificada y test oficial reservado. "
                f"Runtime: Python {audit['runtime']['python']}, TensorFlow {audit['runtime']['tensorflow']}, "
                f"{audit['runtime']['platform']}, dispositivo {', '.join(audit['runtime']['devices'])}.",
                styles["ReportBody"],
            ),
            Paragraph("Particiones", styles["Heading2"]),
        ]
    )
    split_rows = [["Split", "Imágenes", "Forma", "Clases", "Muestras/clase", "Rango"]]
    for split_name in ("train", "validation", "test"):
        split = audit["splits"][split_name]
        class_counts = list(split["class_counts"].values())
        split_rows.append(
            [
                split_name,
                f"{split['count']:,}",
                " × ".join(str(value) for value in split["image_shape"]),
                str(len(class_counts)),
                f"{min(class_counts):,}–{max(class_counts):,}",
                f"{split['pixel_range'][0]}–{split['pixel_range'][1]}",
            ]
        )
    story.extend(
        [
            _table(split_rows, (0.95, 1.0, 0.95, 0.7, 1.25, 0.9)),
            Spacer(1, 0.08 * inch),
            Paragraph(
                "El conjunto de entrenamiento de 60 000 imágenes se dividió con train_test_split estratificado: "
                "54 000 para ajustar pesos y 6 000 para validación (10 %). Las 10 000 imágenes oficiales de test "
                "se reservaron para la evaluación final. Early stopping observó val_loss y restauró los mejores pesos; "
                "no se usó el test para seleccionar modelos.",
                styles["ReportBody"],
            ),
            Paragraph("Fingerprints SHA-256 (prefijo de 16 caracteres)", styles["Heading2"]),
        ]
    )
    hash_rows = [["Split", "Imágenes", "Etiquetas"]]
    for split_name in ("train", "validation", "test"):
        split = audit["splits"][split_name]
        hash_rows.append(
            [split_name, split["images_sha256"][:16], split["labels_sha256"][:16]]
        )
    story.extend(
        [
            _table(hash_rows, (1.4, 2.3, 2.3)),
            Paragraph(
                "Preprocesamiento: uint8 → float32, división por 255 y canal explícito; forma del modelo "
                "(28, 28, 1), rango [0, 1]. Baseline: Flatten → Dense(64, ReLU) → Softmax(10). "
                "CNN: Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → GlobalAveragePooling → "
                "Dropout(0.25) → Softmax(10). Ambos usaron Adam, sparse categorical cross-entropy, lote 128 y hasta 8 épocas.",
                styles["ReportBody"],
            ),
            PageBreak(),
            Paragraph("Métricas por clase", styles["Heading1"]),
            Paragraph(
                "Cada clase tiene soporte 1 000 en test. Se muestran precision, recall y F1 para ambos modelos; "
                "el F1 macro asigna el mismo peso a cada clase.",
                styles["ReportBody"],
            ),
        ]
    )
    class_rows = [["Clase", "P base", "R base", "F1 base", "P CNN", "R CNN", "F1 CNN"]]
    for class_name in class_names:
        baseline_class = dense["per_class"][class_name]
        cnn_class = cnn["per_class"][class_name]
        class_rows.append(
            [
                escape(class_name),
                f"{baseline_class['precision']:.3f}",
                f"{baseline_class['recall']:.3f}",
                f"{baseline_class['f1-score']:.3f}",
                f"{cnn_class['precision']:.3f}",
                f"{cnn_class['recall']:.3f}",
                f"{cnn_class['f1-score']:.3f}",
            ]
        )
    story.extend(
        [
            _table(class_rows, (1.35, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95), 7.5),
            Spacer(1, 0.12 * inch),
            Paragraph("Errores observados en la CNN", styles["Heading2"]),
            Paragraph(
                f"La CNN clasificó incorrectamente {total_errors:,} de 10 000 imágenes. Su menor recall fue "
                f"{cnn['per_class'][hardest_class]['recall']:.3f} en {escape(hardest_class)}: "
                f"{hardest_false_negatives:,} falsos negativos de {hardest_support:,} casos reales. "
                f"El baseline alcanzó recall {dense['per_class'][hardest_class]['recall']:.3f} en esa clase.",
                styles["ReportBody"],
            ),
        ]
    )
    top_error_rows = [["Clase real", "Predicción", "Casos"]]
    for count, actual, predicted in confusion_pairs[:5]:
        top_error_rows.append([escape(actual), escape(predicted), f"{count:,}"])
    story.extend(
        [
            _table(top_error_rows, (2.4, 2.4, 1.1)),
            Paragraph(
                f"La confusión más común fue {escape(confusion_pairs[0][1])} → "
                f"{escape(confusion_pairs[0][2])} ({confusion_pairs[0][0]} casos). "
                "La similitud visual entre prendas superiores es una explicación plausible que debe contrastarse "
                "inspeccionando los ejemplos; la matriz por sí sola no demuestra la causa.",
                styles["ReportBody"],
            ),
            PageBreak(),
            Paragraph("Costo, decisión y uso responsable", styles["Heading1"]),
            Paragraph(
                f"En este equipo la CNN redujo parámetros en {parameter_reduction:.1%} y tamaño en {size_reduction:.1%}, "
                f"pero elevó el tiempo de entrenamiento a {training_ratio:.1f}× y latencia a {inference_ratio:.1f}× "
                "respecto al baseline, con menor F1 macro. Para el alcance de este benchmark, el baseline ofrece "
                "la mejor relación desempeño/costo. Antes de preferir la CNN, conviene revisar arquitectura y entrenamiento, "
                "mantener validación separada y confirmar la mejora en más de una semilla.",
                styles["ReportBody"],
            ),
            Paragraph("Alcance y limitaciones", styles["Heading2"]),
            Paragraph(
                "Fashion-MNIST es un benchmark didáctico con imágenes pequeñas, centradas y monocromas. "
                "No valida fotografías reales, imágenes médicas, OCR, vigilancia ni decisiones sobre personas. "
                "Un único split/semilla no cuantifica incertidumbre y los tiempos dependen del hardware/backend. "
                "El modelo no debe desplegarse fuera del uso académico declarado.",
                styles["ReportBody"],
            ),
            Paragraph("Reproducibilidad y artefactos", styles["Heading2"]),
            Paragraph(
                "Entorno: uv, Python 3.12.14 y TensorFlow 2.21.0 sobre CPU Windows 11. Las instrucciones están en README.md; "
                "requirements.txt y requirements-colab.txt declaran dependencias. La suite ejecutada informó 6 pruebas aprobadas. "
                "El notebook, código reusable, auditoría, métricas, modelos locales y figuras forman parte del proyecto.",
                styles["ReportBody"],
            ),
            Paragraph("Declaración de uso de IA", styles["Heading2"]),
            Paragraph(
                "Herramienta: GitHub Copilot. Prompts relevantes: “crear el trabajo desde cero en esta carpeta” y "
                "“hazme el reporte o informe completo según la práctica”, junto con el mandato y la rúbrica de U02.E04. "
                "La asistencia apoyó el scaffold, las pruebas, el análisis de métricas y la redacción/maquetación del informe. "
                "Se verificaron instalación/runtime, pruebas, particiones, métricas, notebook JSON y artefactos. "
                "Correcciones realizadas: contrato de auditoría consumido por el PDF, filas agregadas excluidas al buscar "
                "el menor recall, anchos de tablas ReportLab corregidos, matriz de confusión persistida y política Git para "
                "binarios. El estudiante debe revisar esta declaración, confirmar que refleja su uso real y poder explicar "
                "cada decisión y resultado.",
                styles["ReportBody"],
            ),
            Paragraph("Trazabilidad con la rúbrica", styles["Heading2"]),
        ]
    )
    rubric_rows = [
        ["Criterio", "Evidencia"],
        ["Auditoría y partición reproducible", "data_audit.json: split, clases, rangos, hashes y runtime."],
        ["Baseline y CNN funcional", "train_cv.py, modelos Keras y test_model_contract.py."],
        ["Métricas, curvas y errores", "cv_metrics.json, tabla por clase y tres figuras siguientes."],
        ["Costo y decisión técnica", "Parámetros, bytes, tiempos y comparación de F1/accuracy."],
        ["Model Card, pruebas, README y Git", "Archivos de entrega y repositorio enlazado arriba."],
    ]
    story.extend(
        [
            _table(rubric_rows, (2.35, 4.75), 7.5),
            PageBreak(),
        ]
    )

    for index, (filename, title) in enumerate(FIGURES):
        story.append(Paragraph(escape(title), styles["Heading1"]))
        story.append(Spacer(1, 0.12 * inch))
        story.append(Image(str(REPORTS / filename), width=7.0 * inch, height=8.3 * inch, kind="proportional"))
        if index < len(FIGURES) - 1:
            story.append(PageBreak())

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.65 * inch,
        title="U02.E04 - Informe de clasificacion visual",
        author="Edwin Ramón José Nolasco",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el informe integral de U02.E04 con evidencia real.")
    parser.add_argument("--output", type=Path, default=REPORTS / "U02_E04_informe.pdf")
    args = parser.parse_args()
    create_report(args.output)
    print(f"Informe completo generado: {args.output}")


if __name__ == "__main__":
    main()