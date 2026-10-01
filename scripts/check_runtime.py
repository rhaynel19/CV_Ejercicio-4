from __future__ import annotations

import platform
import sys


def main() -> None:
    print("Python/OS:", sys.version.split()[0], platform.platform())
    try:
        import tensorflow as tf
    except Exception as error:
        raise SystemExit(
            "TensorFlow no pudo importarse. Use Python 3.12 en Linux, WSL2 o Colab; "
            "revise requirements.txt y no continúe al entrenamiento hasta resolverlo. "
            f"Detalle: {type(error).__name__}: {error}"
        ) from error
    print("TensorFlow:", tf.__version__)
    print("Dispositivos:", [device.name for device in tf.config.list_physical_devices()])
    print("GPU:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()