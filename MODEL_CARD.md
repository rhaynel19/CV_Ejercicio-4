# Model Card · Fashion-MNIST

## Identificación

- Ejercicio: U02.E04, INF-8239 Ciencia de Datos II.
- Autor: Edwin Ramón José Nolasco.
- Modelos: baseline denso y CNN pequeña implementados con Keras.
- Estado: ejecución completada en Windows 11, Python 3.12.14, TensorFlow 2.21.0, CPU.

## Uso previsto y fuera de alcance

Uso didáctico para comparar clasificadores en las diez categorías de Fashion-MNIST. No se valida para producción, fotografías de prendas del mundo real, inferencia de atributos personales ni decisiones comerciales u operativas.

## Datos y particiones

Fashion-MNIST contiene 60 000 imágenes de entrenamiento y 10 000 de prueba, escala de grises de 28 x 28 píxeles y diez clases. Fuente: Keras `fashion_mnist`, con referencia al [repositorio Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). Se fija la semilla 42 y se hace partición estratificada 90/10 del entrenamiento para train/validación. El test oficial queda reservado para evaluación final. `reports/data_audit.json` registra conteos, formas, rango, fingerprints SHA-256 y runtime.

## Preprocesamiento y arquitectura

Las imágenes se convierten a `float32`, se dividen por 255 y reciben un canal: `(28, 28, 1)`, rango `[0, 1]`. Baseline: flatten, densa de 64 unidades ReLU y softmax de diez clases. CNN: convoluciones de 32 y 64 filtros, max pooling, global average pooling, dropout 0.25 y softmax de diez clases.

## Evaluación y costo

Reportar accuracy, macro-F1 y precision/recall/F1 por clase; comparar parámetros, tamaño serializado, duración de entrenamiento y mediana de latencia de inferencia por imagen. La evaluación final es el test oficial y la selección de pesos usa validación.

## Resultados observados

Ejecución con semilla 42: 54 000 imágenes de entrenamiento, 6 000 de validación y 10 000 de prueba. El baseline obtuvo accuracy 0.8618 y F1 macro 0.8605, con 50 890 parámetros, 633 964 bytes, 23.7 s de entrenamiento y 0.0425 ms/imagen de inferencia. La CNN obtuvo accuracy 0.7918 y F1 macro 0.7891, con 19 466 parámetros, 268 447 bytes, 124.6 s de entrenamiento y 0.1419 ms/imagen.

En esta ejecución la clase con menor recall de la CNN fue `Shirt`, con 0.424. Consultar `reports/cv_metrics.json` para el reporte por clase completo y `reports/data_audit.json` para la configuración de datos/runtime.

## Decisión técnica

Para este split, el baseline es preferible: supera a la CNN por 0.0714 en F1 macro y es más rápido al entrenar e inferir. La CNN ocupa menos espacio y tiene menos parámetros, pero esa ventaja no compensa su menor desempeño predictivo en esta ejecución. No se afirma que la diferencia sea universal: corresponde a una semilla, partición y entorno concretos.

## Limitaciones y supervisión

Imágenes pequeñas, centradas, monocromas y con fondos controlados; no representan fotografías cotidianas. Las clases pueden compartir siluetas. Un solo split y semilla no miden toda la variabilidad. Los tiempos dependen del hardware. Revisar métricas por clase, matriz de confusión y errores visuales antes de cualquier conclusión; no transferir el resultado a otros dominios.

La ejecución se realizó en CPU; TensorFlow no detectó GPU en Windows nativo. Las figuras y métricas están guardadas en `reports/`, y los modelos serializados en `models/`.
