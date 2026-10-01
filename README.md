# U02.E04 · Clasificador visual con CNN y Model Card

**Asignatura:** INF-8239 Ciencia de Datos II  
**Unidad:** 02 · Procesamiento de Lenguaje Natural  
**Autor:** Edwin Ramón José Nolasco  
**Código:** U02.E04

## Propósito

Comparar un baseline denso con una CNN sobre Fashion-MNIST. El flujo audita los datos, fija una partición reproducible, evalúa métricas por clase, registra costo computacional y genera figuras e informe desde resultados reales.

## Requisitos

Se requiere Python 3.12. El proyecto se ejecutó en Windows 11 con TensorFlow 2.21 sobre CPU; TensorFlow no ofrece GPU en Windows nativo para esta versión. Linux, WSL2 o Google Colab son alternativas para GPU. La primera corrida descarga Fashion-MNIST y necesita Internet.

### Entorno reproducible con uv

```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run pytest -q
```

### Entrenamiento e informe

```bash
uv run python scripts/train_cv.py --epochs 8
uv run python scripts/build_submission_pdf.py
```

Para ejecutar el notebook completo y conservar los outputs en el `.ipynb`:

```bash
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/U02_E04_clasificador_visual.ipynb
uv run python scripts/normalize_notebook_metadata.py notebooks/U02_E04_clasificador_visual.ipynb
uv run python scripts/build_submission_pdf.py
```

Alternativa con pip: `python -m pip install -r requirements.txt`, y luego ejecute los mismos scripts con `python`.

## Diseño experimental

Se fija la semilla 42. El conjunto oficial de prueba queda reservado; el 10 % del entrenamiento se separa estratificadamente para validación. Early stopping observa validación y no utiliza las etiquetas del test. Baseline y CNN reciben el mismo preprocesamiento: conversión a `float32`, escalado a `[0, 1]` y canal explícito.

El script genera:

- `reports/data_audit.json`: partición, conteos de clase, dimensiones, rango de píxeles, fingerprints SHA-256 y runtime.
- `reports/cv_metrics.json`: accuracy, macro-F1, métricas por clase, parámetros, tamaño, duración de entrenamiento e inferencia.
- `reports/training_curves.png`, `reports/confusion_cnn.png` y `reports/cnn_errors.png`.
- `models/dense.keras`, `models/cnn.keras` y `models/best_cnn.keras`.
- `reports/U02_E04_informe.pdf`, si ya existen métricas y las tres figuras.

Las métricas y figuras no se incluyen como resultados de muestra. Se crean al completar entrenamiento real. Los tiempos dependen del hardware.

## Notebook y evaluación

Abra `notebooks/U02_E04_clasificador_visual.ipynb`, elija el kernel del entorno `.venv` y ejecute todas las celdas. El notebook lanza el mismo script de entrenamiento, carga sus JSON y presenta la evidencia visual. Guarde el notebook con outputs solo después de completar la ejecución en un runtime TensorFlow funcional. Antes de entregar, normalice los metadatos requeridos por la plantilla:

```bash
uv run python scripts/normalize_notebook_metadata.py notebooks/U02_E04_clasificador_visual.ipynb
```

Las pruebas de datos son independientes de TensorFlow. La prueba del contrato de salida de la CNN se omite cuando TensorFlow no está instalado.

## Model Card y uso responsable

Complete `MODEL_CARD.md` con las métricas observadas. Fashion-MNIST es un benchmark educativo de prendas pequeñas y centradas; sus resultados no validan fotografías reales, imágenes médicas, OCR, vigilancia ni decisiones sobre personas.

## Repositorio, PDF e IA

El PDF detecta el remoto `origin` si existe. Sin remoto, lo declara pendiente en lugar de inventar un enlace. Publique el repositorio, configure el remoto y regenere el PDF. La declaración de IA incluida en el informe es una base: el estudiante debe corregirla para reflejar herramientas, prompts, verificaciones y correcciones efectivamente realizadas.

Para registrar el trabajo:

```bash
git init
git add .
git commit -m "feat: build reproducible U02.E04 computer vision project"
git remote add origin https://github.com/USUARIO/REPOSITORIO.git
git push -u origin main
```

Los JSON, figuras y PDF dentro de `reports/` no están excluidos por `.gitignore`; incorpórelos después de producir evidencia real. Los modelos serializados y datos descargados se excluyen por tamaño.

No entregue esta plantilla como corrida completa hasta ejecutar el notebook y generar las métricas, figuras y PDF en un runtime compatible.
