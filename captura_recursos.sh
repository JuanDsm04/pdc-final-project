#!/bin/bash
# Captura de pantalla del monitor de recursos (Linux con Hyprland y grim).
# Abre una ventana con la fase actual, nvidia-smi y el uso por núcleo (mpstat),
# toma la captura de esa ventana y la cierra.
# Uso: ./captura_recursos.sh evidencias/nombre.png [segundos de espera, 5 por defecto]
set -euo pipefail
DESTINO=$1
ESPERA=${2:-5}
ALTO_MAX=600  # el contenido ocupa unas 42 líneas; el resto de la ventana queda vacío
CLASE=pdc-monitor-$$
REPO=$(cd "$(dirname "$0")" && pwd)

kitty --class "$CLASE" -o font_size=8.5 \
    sh -c "cd '$REPO' && watch -t -n 1 'date \"+%Y-%m-%d %H:%M:%S\"; cat results/fase_actual.txt; echo; nvidia-smi; mpstat -P ALL 1 1 | grep -v Average | tail -n +3'" &
PID_KITTY=$!
trap 'kill $PID_KITTY 2>/dev/null || true' EXIT

geometria() {
    hyprctl clients -j | python3 -c "
import json, sys
for c in json.load(sys.stdin):
    if c['class'] == '$CLASE':
        print(f\"{c['at'][0]},{c['at'][1]} {c['size'][0]}x{min(c['size'][1], $ALTO_MAX)}\")"
}

for _ in $(seq 50); do
    [ -n "$(geometria)" ] && break
    sleep 0.1
done
sleep "$ESPERA"  # tiempo para que watch muestre al menos una lectura completa
mkdir -p "$(dirname "$DESTINO")"
grim -g "$(geometria)" "$DESTINO"
echo "captura guardada en $DESTINO"
