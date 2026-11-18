"""Registro del uso de CPU por núcleo y de GPU durante la ejecución de los notebooks.

Un hilo toma una muestra por segundo y la etiqueta con la fase activa. Al detenerse
deja el log completo, un resumen por fase y una gráfica en la carpeta de evidencias.
"""
import csv
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

CONSULTA_GPU = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
CAMPOS_GPU = ["gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "gpu_potencia_w", "gpu_temp_c"]
FASES_CON_VOLCADO = ("preprocesamiento", "entrenamiento", "inferencia", "extraccion", "ajuste")
SIN_FASE = "sin fase"


def _a_numero(texto):
    try:
        return float(texto)
    except ValueError:
        return None


class MonitorRecursos:
    def __init__(self, parte, modo, results_dir, evidencias_dir, intervalo=1.0, espera_volcado=5.0):
        self.parte, self.modo = parte, modo
        self.results_dir, self.evidencias_dir = Path(results_dir), Path(evidencias_dir)
        self.intervalo, self.espera_volcado = intervalo, espera_volcado
        self.csv_path = self.results_dir / f"recursos_{parte}_{modo}.csv"
        self.resumen_path = self.results_dir / f"recursos_resumen_{parte}_{modo}.csv"
        self.figura_path = self.evidencias_dir / f"recursos_{parte}_{modo}.png"
        self.fase_path = self.results_dir / "fase_actual.txt"
        self.n_nucleos = psutil.cpu_count(logical=True)
        self.hay_gpu = shutil.which("nvidia-smi") is not None
        self._fase = SIN_FASE
        self._inicio_fase = time.perf_counter()
        self._volcadas = set()
        self._gpu = dict.fromkeys(CAMPOS_GPU)
        self._parar = threading.Event()
        self._hilos = []
        self._proc_gpu = None

    def iniciar(self):
        for carpeta in (self.results_dir, self.evidencias_dir):
            carpeta.mkdir(parents=True, exist_ok=True)
        self._archivo = open(self.csv_path, "w", newline="")
        self._escritor = csv.writer(self._archivo)
        self._escritor.writerow(
            ["t_s", "hora", "fase", "cpu_total"] + [f"cpu_{i}" for i in range(self.n_nucleos)]
            + ["ram_gb"] + CAMPOS_GPU
        )
        psutil.cpu_percent(percpu=True)
        self._t0 = time.perf_counter()
        if self.hay_gpu:
            # nvidia-smi usa NVML: ve la GPU aunque CUDA_VISIBLE_DEVICES la oculte al proceso
            self._proc_gpu = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={CONSULTA_GPU}", "--format=csv,noheader,nounits",
                 "-lms", str(int(self.intervalo * 1000))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._lanzar(self._leer_gpu)
        self._lanzar(self._muestrear)
        self._escribir_fase()
        return self

    def _lanzar(self, funcion):
        hilo = threading.Thread(target=funcion, daemon=True)
        hilo.start()
        self._hilos.append(hilo)

    def _leer_gpu(self):
        for linea in self._proc_gpu.stdout:
            valores = [_a_numero(v.strip()) for v in linea.split(",")]
            if len(valores) == len(CAMPOS_GPU):
                self._gpu = dict(zip(CAMPOS_GPU, valores))

    def _muestrear(self):
        while not self._parar.wait(self.intervalo):
            por_nucleo = psutil.cpu_percent(percpu=True)
            fase = self._fase
            self._escritor.writerow(
                [round(time.perf_counter() - self._t0, 2), time.strftime("%H:%M:%S"), fase,
                 round(sum(por_nucleo) / len(por_nucleo), 1)] + por_nucleo
                + [round(psutil.virtual_memory().used / 1e9, 2)] + [self._gpu[c] for c in CAMPOS_GPU]
            )
            self._archivo.flush()
            self._volcar_nvidia_smi(fase)

    def _volcar_nvidia_smi(self, fase):
        if (not self.hay_gpu or fase in self._volcadas or not fase.startswith(FASES_CON_VOLCADO)
                or time.perf_counter() - self._inicio_fase < self.espera_volcado):
            return
        self._volcadas.add(fase)
        salida = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        nombre = re.sub(r"[^a-z0-9]+", "-", fase.lower()).strip("-")
        destino = self.evidencias_dir / f"nvidia-smi_{self.parte}_{self.modo}_{nombre}.txt"
        destino.write_text(f"Fase: {fase}\nHora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{salida}")

    def _escribir_fase(self):
        self.fase_path.write_text(f"{time.strftime('%H:%M:%S')}  {self.parte} ({self.modo}): {self._fase}\n")

    def marcar(self, fase):
        """Cambia la fase activa y devuelve la anterior."""
        anterior = self._fase
        self._fase = fase
        self._inicio_fase = time.perf_counter()
        self._escribir_fase()
        return anterior

    @contextmanager
    def fase(self, nombre):
        anterior = self.marcar(nombre)
        try:
            yield
        finally:
            self.marcar(anterior)

    def detener(self):
        self._parar.set()
        if self._proc_gpu is not None:
            self._proc_gpu.terminate()
        for hilo in self._hilos:
            hilo.join(timeout=5)
        self._archivo.close()
        self.fase_path.write_text(f"{time.strftime('%H:%M:%S')}  {self.parte} ({self.modo}): terminado\n")

    def muestras(self):
        import pandas as pd
        return pd.read_csv(self.csv_path)

    def resumen(self, umbral=50.0):
        """Uso de CPU y GPU por fase.

        `nucleos_equivalentes` es la carga total expresada en núcleos al 100 %; no depende de que el
        sistema operativo mueva un hilo de un núcleo a otro. `nucleos_activos` cuenta los núcleos por
        encima del umbral.
        """
        df = self.muestras()
        nucleos = [c for c in df.columns if re.fullmatch(r"cpu_\d+", c)]
        df["nucleos_equivalentes"] = df[nucleos].sum(axis=1) / 100.0
        df["nucleos_activos"] = (df[nucleos] > umbral).sum(axis=1)
        df["nucleo_mas_cargado"] = df[nucleos].max(axis=1)
        resumen = (
            df[df["fase"] != SIN_FASE].groupby("fase", sort=False)
            .agg(segundos=("t_s", "size"), cpu_total_medio=("cpu_total", "mean"),
                 nucleos_equivalentes=("nucleos_equivalentes", "mean"),
                 nucleos_activos_medio=("nucleos_activos", "mean"),
                 nucleo_mas_cargado_medio=("nucleo_mas_cargado", "mean"),
                 ram_max_gb=("ram_gb", "max"), gpu_util_media=("gpu_util", "mean"),
                 gpu_util_max=("gpu_util", "max"), gpu_mem_max_mb=("gpu_mem_mb", "max"))
            .reset_index()
        )
        resumen["segundos"] = resumen["segundos"] * self.intervalo
        resumen = resumen.round(1)
        resumen.to_csv(self.resumen_path, index=False)
        return resumen

    def graficar(self, max_columnas=1500, titulo=None):
        """Mapa de calor del uso por núcleo, con el uso y la memoria de la GPU debajo."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        tinta, tinta2, rejilla, fondo, azul = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb", "#2a78d6"
        mapa = LinearSegmentedColormap.from_list(
            "azul", [fondo, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

        df = self.muestras()
        nucleos = [c for c in df.columns if re.fullmatch(r"cpu_\d+", c)]
        paso = max(1, int(np.ceil(len(df) / max_columnas)))
        grupo = np.arange(len(df)) // paso
        cpu = df[nucleos].groupby(grupo).mean().to_numpy().T
        t = df["t_s"].groupby(grupo).max().to_numpy()      # cada muestra cubre el intervalo que termina en t
        t_fin = float(df["t_s"].iloc[-1])
        en_minutos = t_fin > 600
        escala = 60.0 if en_minutos else 1.0

        fig, ejes = plt.subplots(
            3, 1, figsize=(13, 7.5), sharex=True, facecolor=fondo,
            gridspec_kw={"height_ratios": [3.2, 1.2, 1.2], "hspace": 0.16})
        ax_cpu, ax_gpu, ax_mem = ejes
        malla = ax_cpu.pcolormesh(
            np.insert(t, 0, 0.0) / escala, np.arange(len(nucleos) + 1), cpu,
            cmap=mapa, vmin=0, vmax=100, shading="flat", rasterized=True)
        ax_cpu.set_yticks(np.arange(len(nucleos)) + 0.5, [str(i) for i in range(len(nucleos))], fontsize=7)
        ax_cpu.set_ylabel("Núcleo lógico", color=tinta2)
        ax_cpu.invert_yaxis()
        barra = fig.colorbar(malla, ax=ejes, pad=0.012, fraction=0.03, aspect=40)
        barra.set_label("Uso del núcleo (%)", color=tinta2)
        barra.outline.set_visible(False)

        gpu_util = df["gpu_util"].groupby(grupo).mean().to_numpy()
        gpu_mem = df["gpu_mem_mb"].groupby(grupo).max().to_numpy()
        ax_gpu.fill_between(t / escala, gpu_util, color=azul, alpha=0.18, linewidth=0)
        ax_gpu.plot(t / escala, gpu_util, color=azul, linewidth=1.4)
        ax_gpu.set_ylim(0, 105)
        ax_gpu.set_ylabel("GPU (%)", color=tinta2)
        ax_mem.fill_between(t / escala, gpu_mem, color=azul, alpha=0.18, linewidth=0)
        ax_mem.plot(t / escala, gpu_mem, color=azul, linewidth=1.4)
        total = df["gpu_mem_total_mb"].dropna()
        ax_mem.set_ylim(0, float(total.iloc[0]) * 1.05 if len(total) else None)
        ax_mem.set_ylabel("Memoria GPU (MiB)", color=tinta2)
        ax_mem.set_xlabel("Tiempo desde el inicio (min)" if en_minutos else "Tiempo desde el inicio (s)",
                          color=tinta2)

        # Límites de fase, con nombre solo en las fases que ocupan espacio suficiente
        cambios = df.index[df["fase"] != df["fase"].shift()].tolist() + [len(df)]
        ultimo_rotulo = -1.0
        for ini, fin in zip(cambios[:-1], cambios[1:]):
            nombre = df["fase"].iloc[ini]
            x0 = df["t_s"].iloc[ini - 1] / escala if ini else 0.0
            x1 = df["t_s"].iloc[fin - 1] / escala
            for ax in ejes if ini else ():
                ax.axvline(x0, color=tinta2, linewidth=0.5, alpha=0.55)
            centro = (x0 + x1) / 2
            ancho_minimo = 0.035 * t_fin / escala
            if nombre != SIN_FASE and x1 - x0 >= ancho_minimo and centro - ultimo_rotulo >= ancho_minimo:
                ax_cpu.text(centro, -0.25, nombre, rotation=35, ha="left", va="bottom",
                            fontsize=7, color=tinta2)
                ultimo_rotulo = centro

        for ax in ejes:
            ax.set_facecolor(fondo)
            ax.tick_params(colors=tinta2, labelsize=8)
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            for lado in ("left", "bottom"):
                ax.spines[lado].set_color("#c3c2b7")
        for ax in (ax_gpu, ax_mem):
            ax.grid(True, axis="y", color=rejilla, linewidth=0.6)
            ax.set_axisbelow(True)
        ax_cpu.set_xlim(0, t_fin / escala)
        ax_cpu.set_title(titulo or f"Uso de recursos: versión {self.parte}, modo {self.modo}",
                         loc="left", fontsize=13, color=tinta, pad=110)
        fig.savefig(self.figura_path, dpi=150, bbox_inches="tight", facecolor=fondo)
        return fig
