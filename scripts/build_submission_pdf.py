from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ("training_curves.png", "confusion_cnn.png", "cnn_errors.png")


def get_repository_url() -> str:
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


def create_pdf(output: Path) -> None:
    audit_path = REPORTS / "data_audit.json"
    metrics_path = REPORTS / "cv_metrics.json"
    missing_data = [path.name for path in (audit_path, metrics_path) if not path.is_file()]
    missing_figures = [name for name in FIGURES if not (REPORTS / name).is_file()]
    if missing_data or missing_figures:
        missing = missing_data + missing_figures
        raise FileNotFoundError(
            "No se genera PDF con evidencia incompleta. Faltan: " + ", ".join(missing)
            + ". Ejecute primero scripts/train_cv.py en un runtime TensorFlow funcional."
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    dense, cnn = metrics["dense"], metrics["cnn"]
    f1_delta = cnn["f1_macro"] - dense["f1_macro"]
    decision = (
        f"La CNN mejora el F1 macro en {f1_delta:+.4f}; para esta ejecución, "
        "su uso se justifica solo si la ganancia compensa el costo adicional medido."
        if f1_delta > 0
        else f"El baseline obtiene un F1 macro {abs(f1_delta):.4f} superior; "
        "esta evidencia no justifica seleccionar la CNN por desempeño."
    )
    repo = get_repository_url()
    repository_text = (
        f'Repositorio: <link href="{escape(repo)}" color="#176B73">{escape(repo)}</link>'
        if repo
        else "Repositorio público: pendiente de publicar y configurar el remoto origin."
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph("Ejercicio 04 · Clasificador visual con CNN y Model Card", styles["Title"]),
        Paragraph("INF-8239 Ciencia de Datos II · Unidad 02 · U02.E04", styles["Normal"]),
        Paragraph("Edwin Ramón José Nolasco · Universidad Autónoma de Santo Domingo", styles["Normal"]),
        Spacer(1, 0.15 * inch),
        Paragraph(repository_text, styles["BodyText"]),
        Paragraph(
            "Fuente: Fashion-MNIST de Keras. Partición estratificada del conjunto de entrenamiento; "
            f"semilla {audit['seed']}, entrenamiento {audit['counts']['train']:,}, "
            f"validación {audit['counts']['validation']:,}, test reservado {audit['counts']['test']:,}.",
            styles["BodyText"],
        ),
        Paragraph(
            f"Runtime: Python {audit['runtime']['python']}, TensorFlow {audit['runtime']['tensorflow']}; "
            f"{audit['runtime']['platform']}; dispositivos: {', '.join(audit['runtime']['devices'])}.",
            styles["BodyText"],
        ),
        Spacer(1, 0.15 * inch),
    ]
    table_rows = [["Modelo", "Accuracy", "F1 macro", "Parámetros", "Bytes", "Entrenamiento", "ms/imagen"]]
    for name, values in (("Baseline denso", dense), ("CNN", cnn)):
        table_rows.append(
            [
                name,
                f"{values['accuracy']:.4f}",
                f"{values['f1_macro']:.4f}",
                f"{values['parameters']:,}",
                f"{values['model_size_bytes']:,}",
                f"{values['train_seconds']:.1f} s",
                f"{values['inference_ms_per_image']:.4f}",
            ]
        )
    table = Table(
        table_rows,
        repeatRows=1,
        colWidths=[width * inch for width in (1.0, 0.68, 0.68, 0.83, 0.8, 0.9, 0.75)],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#184E5A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C7C9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF4F2")]),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 0.12 * inch),
            Paragraph("Decisión técnica", styles["Heading2"]),
            Paragraph(escape(decision), styles["BodyText"]),
            Paragraph(
                f"La clase de menor recall en CNN fue {min(audit['class_names'], key=lambda name: cnn['per_class'][name]['recall'])} "
                "(ver tabla y matriz). La conclusión aplica a Fashion-MNIST; no valida fotografías reales ni otros dominios.",
                styles["BodyText"],
            ),
            Paragraph("Uso de IA y verificación", styles["Heading2"]),
            Paragraph(
                "Se utilizó GitHub Copilot como apoyo para estructurar el proyecto, proponer pruebas y revisar el código. "
                "Prompt relevante (resumido): crear desde cero la entrega U02.E04 conforme al mandato, con notebook, "
                "baseline/CNN, evaluación, costos, Model Card y PDF. Se verificaron los contratos de datos con pytest; "
                "el entrenamiento y las métricas de este informe provienen de los archivos de esta corrida. "
                "Los cambios asistidos incluyen partición estratificada, auditoría con fingerprints, métricas por clase, "
                "registro de costo, figuras, pruebas de contrato y generación de informe. El autor debe declarar también "
                "cualquier otra herramienta, prompt, verificación o corrección propia y explicar el código.",
                styles["BodyText"],
            ),
            PageBreak(),
            Paragraph("Métricas por clase · CNN", styles["Heading1"]),
        ]
    )
    class_rows = [["Clase", "Precisión", "Recall", "F1", "Soporte"]]
    for class_name in audit["class_names"]:
        class_metrics = cnn["per_class"][class_name]
        class_rows.append(
            [
                class_name,
                f"{class_metrics['precision']:.3f}",
                f"{class_metrics['recall']:.3f}",
                f"{class_metrics['f1-score']:.3f}",
                f"{int(class_metrics['support']):,}",
            ]
        )
    class_table = Table(
        class_rows,
        repeatRows=1,
        colWidths=[width * inch for width in (2.3, 1.0, 1.0, 1.0, 1.0)],
    )
    class_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#184E5A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C7C9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF4F2")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            class_table,
            Spacer(1, 0.16 * inch),
            Paragraph("Límites", styles["Heading2"]),
            Paragraph(
                "Fashion-MNIST contiene imágenes pequeñas, centradas y en escala de grises. "
                "La variación de fondo, resolución e iluminación de fotografías cotidianas no está representada. "
                "Inspeccione la matriz y errores antes de decidir; no extrapole el resultado a uso operativo.",
                styles["BodyText"],
            ),
            PageBreak(),
        ]
    )
    for index, filename in enumerate(FIGURES):
        story.append(Paragraph(escape(filename.replace("_", " ").replace(".png", "").title()), styles["Heading1"]))
        image = Image(str(REPORTS / filename), width=6.6 * inch, height=4.9 * inch, kind="proportional")
        story.append(image)
        if index < len(FIGURES) - 1:
            story.append(PageBreak())

    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    ).build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un PDF breve desde métricas y figuras reales.")
    parser.add_argument("--output", type=Path, default=REPORTS / "U02_E04_informe.pdf")
    args = parser.parse_args()
    create_pdf(args.output)
    print(f"PDF generado: {args.output}")


if __name__ == "__main__":
    main()