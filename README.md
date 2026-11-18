# Proyecto 3: Reconocimiento de ASL con CUDA

CC3069 Computación Paralela y Distribuida, Universidad del Valle de Guatemala.

Toma la solución del Laboratorio 3 de Data Science (clasificador del alfabeto ASL con CNN), mide su desempeño secuencial en CPU y la acelera con CUDA.

## Integrantes

- Juan Diego Solís Martínez - 23720
- Víctor Manuel Pérez Chávez - 23731

## Estructura

| Ruta | Contenido |
|---|---|
| `proyecto3_asl_secuencial.ipynb` | Versión secuencial: solución del laboratorio cronometrada en CPU |
| `proyecto3_asl_cuda.ipynb` | Versión paralela: preparación de datos en CPU multinúcleo, entrenamiento en GPU con TensorFlow y kernels propios con Numba CUDA; calcula el speedup |
| `monitor_recursos.py` | Registro del uso de cada núcleo del CPU y de la GPU durante la ejecución |
| `artifacts/splits.csv` | Partición train/val/test (70/15/15) del laboratorio |
| `fotos_grupo/` | Fotos de señas propias usadas en las pruebas finales |
| `results/` | Tiempos y hardware que genera el notebook |
| `figures/` | Gráficas que exporta el notebook |
| `evidencias/` | Capturas, gráficas y volcados de `nvidia-smi` del uso de recursos durante la ejecución |
| `archive/` | Dataset (no se versiona) |

## Dataset

[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87,000 imágenes, 29 clases). El notebook busca `archive/asl_alphabet_train/asl_alphabet_train/`. Si la carpeta no existe, lo descarga con `kagglehub`; si Kaggle pide autenticación, hay que configurar las credenciales (`KAGGLE_USERNAME` y `KAGGLE_KEY`).

## Ejecución

### Local

TensorFlow 2.21 requiere Python 3.10 a 3.13. En Linux, `requirements.txt` instala además las librerías de CUDA 12 dentro del entorno (`tensorflow[and-cuda]` y `numba-cuda[cu12]`), así que solo hace falta el driver de NVIDIA.

```bash
# Linux, con uv
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 proyecto3_asl_secuencial.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 proyecto3_asl_cuda.ipynb
```

El notebook CUDA lee los tiempos que deja el secuencial en `results/`, así que se ejecuta después y en la misma máquina. Cada notebook corre en su propio proceso para que las mediciones de CPU y GPU no se mezclen.

Dos detalles del entorno en Linux, ya resueltos en el código:

- `numpy` se limita a versiones menores que 2.5, porque `numba-cuda` 0.30 todavía usa `np.row_stack`.
- TensorFlow 2.21 no encuentra por sí solo el cuSOLVER que instala pip y, sin avisar con un error, deja de usar la GPU. El notebook CUDA carga esa librería antes de importar TensorFlow.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook proyecto3_asl_secuencial.ipynb
```

### Google Colab

1. Abrir `proyecto3_asl_secuencial.ipynb` en Colab con un entorno de ejecución de GPU.
2. Ejecutar todas las celdas. La celda de entorno clona el repositorio y descarga el dataset si hace falta.

La versión secuencial oculta la GPU (`CUDA_VISIBLE_DEVICES=-1`), así que toda la ejecución ocurre en CPU. Se recomiendan al menos 8 GB de RAM: el preprocesamiento a memoria ocupa unos 1.1 GB en benchmark y 3.0 GB en completo.

## Configuración (celda 0)

| Parámetro | Valor | Uso |
|---|---|---|
| `MODO` | `"benchmark"` | Submuestra y épocas fijas, con repeticiones. `"completo"` reproduce el laboratorio en una corrida |
| `N_REPETICIONES` | 3 | Repeticiones de las fases medidas (solo en benchmark) |
| `IMGS_POR_CLASE_BENCHMARK` | 800 | Imágenes de entrenamiento por clase |
| `EPOCAS_BENCHMARK` | 5 | Épocas fijas, sin EarlyStopping |
| `N_JOBS` | 1 | Procesos para la extracción HOG y los ensambles (1 = secuencial). En el notebook CUDA es el número de núcleos lógicos |

Los mismos parámetros se pueden cambiar sin editar el notebook, con las variables de entorno `ASL_MODO`, `ASL_REPETICIONES`, `ASL_IMGS_POR_CLASE`, `ASL_EPOCAS` y, en el notebook CUDA, `ASL_N_JOBS`. `ASL_SALIDA` indica la carpeta donde se crean `results/`, `figures/` y `evidencias/` (por defecto, la del repositorio). Por ejemplo, para la corrida completa:

```bash
ASL_MODO=completo .venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output-dir ejecuciones \
    --output proyecto3_asl_secuencial_completo.ipynb proyecto3_asl_secuencial.ipynb
```

## Resultados que genera

Todos los archivos llevan el modo (`benchmark` o `completo`) en el nombre, para que una corrida no sobrescriba a la otra. Cada notebook escribe sus propios archivos con `secuencial` o `cuda` en el nombre; abajo se listan los del secuencial.

- `results/hardware_secuencial_<modo>.json`: máquina, núcleos, RAM, GPU y versiones.
- `results/tiempos_secuencial_<modo>.csv`: cada medición individual.
- `results/resumen_secuencial_<modo>.csv` y `results/tiempo_por_epoca_secuencial_<modo>.csv`: media y desviación por fase.
- `results/metricas_secuencial_<modo>.json`: exactitud de cada modelo en validación y test.
- `results/recursos_secuencial_<modo>.csv` y `results/recursos_resumen_secuencial_<modo>.csv`: uso por núcleo y de GPU, muestra a muestra y resumido por fase.
- `results/fase_actual.txt`: fase que se está ejecutando en este momento.
- `figures/tiempos_secuencial_<modo>.png`: tiempo por fase.
- `evidencias/recursos_secuencial_<modo>.png` y `evidencias/nvidia-smi_secuencial_<modo>_<fase>.txt`: uso de recursos en el tiempo y estado de la GPU durante cada fase.

Además, el notebook CUDA genera:

- `results/comparacion_<modo>.csv`, `figures/comparacion_tiempos_<modo>.png` y `figures/speedup_<modo>.png`: tiempo secuencial, tiempo CUDA y speedup por fase.
- `results/exactitud_<modo>.csv`: exactitud de cada modelo en ambas versiones.
- `results/kernels_numba_<modo>.csv`: grid, bloque, hilos y tiempo de cada kernel propio.
- `results/escalabilidad_preprocesamiento.csv` y `figures/escalabilidad_preprocesamiento.png`: speedup del preprocesamiento según el número de procesos.
- `results/origen_speedup_benchmark.csv` y `figures/origen_speedup_benchmark.png`: cuánto del speedup del entrenamiento viene de tener los datos en memoria y cuánto de la GPU.
