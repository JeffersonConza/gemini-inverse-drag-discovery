#!/bin/bash
set -e

# 1. Validación de Entorno Virtual
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: El entorno virtual no está activado."
    echo "Por favor, ejecuta: source venv/bin/activate"
    exit 1
fi

# 2. Inicio del temporizador
start_time=$(date +%s)

# Asegurar directorios
mkdir -p data docs

echo "🚀 Iniciando simulación de telemetría (Ground Truth)..."
python3 src/simulador.py --masa 500 --cd 0.3 --angulo 85 --salida data/vuelo.csv

echo "🤖 Invocando a Gemini CLI para auditar la física del vuelo..."
if ! gemini -p "Actúa como un analista SciML. Analiza el archivo @data/vuelo.csv. 
Verifica si la curva de velocidad ('vy') muestra los efectos de la resistencia atmosférica (drag). 
Redacta un reporte breve indicando si estos datos tienen suficiente varianza no lineal 
para entrenar una PINN que descubra el coeficiente de arrastre." > docs/reporte_cd.txt; then
    echo "❌ ERROR: Falló la auditoría con Gemini CLI."
    exit 1
fi

# Verificar que el reporte no esté vacío (evitar falsos positivos por errores de red)
if [ ! -s docs/reporte_cd.txt ]; then
    echo "❌ ERROR: El reporte generado está vacío. Comprueba la conexión o cuotas de la API."
    exit 1
fi

# 3. Cálculo de tiempo total
end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "✅ Pipeline completado en ${execution_time}s. Revisa docs/reporte_cd.txt"
