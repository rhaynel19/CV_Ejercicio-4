from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def normalize_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            language = "python"
        else:
            language = "markdown"
        cell.metadata["id"] = cell.id
        cell.metadata["language"] = language
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3.12 (U02.E04)",
        "language": "python",
        "name": "inf8239-u02-e04",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Añade metadatos de celda requeridos al notebook.")
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    normalize_notebook(args.notebook)
    print(f"Metadatos normalizados: {args.notebook}")


if __name__ == "__main__":
    main()