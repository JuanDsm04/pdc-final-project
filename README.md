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
| `artifacts/splits.csv` | Partición train/val/test (70/15/15) del laboratorio |
| `fotos_grupo/` | Fotos de señas propias usadas en las pruebas finales |
| `results/` | Tiempos y hardware que genera el notebook |
| `figures/` | Gráficas que exporta el notebook |
| `evidencias/` | Capturas del uso de recursos durante la ejecución |
| `archive/` | Dataset (no se versiona) |

## Dataset

[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87,000 imágenes, 29 clases). El notebook busca `archive/asl_alphabet_train/asl_alphabet_train/`. Si la carpeta no existe, lo descarga con `kagglehub`; si Kaggle pide autenticación, hay que configurar las credenciales (`KAGGLE_USERNAME` y `KAGGLE_KEY`).

## Ejecución

### Local

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
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
| `N_JOBS` | 1 | Procesos para la extracción HOG y los ensambles (1 = secuencial) |

## Resultados que genera

- `results/hardware.json`: máquina, núcleos, RAM, GPU y versiones.
- `results/tiempos_secuencial.csv`: cada medición individual.
- `results/resumen_secuencial.csv` y `results/tiempo_por_epoca_secuencial.csv`: media y desviación por fase.
- `figures/tiempos_secuencial.png`: tiempo por fase.
