import argparse
import csv
import numpy as np
from integrador import rk4_step, cinematica_cohete

def simular(masa, cd, angulo, salida):
    """
    Simula la trayectoria de un cohete y genera telemetría con ruido instrumental.
    """
    t, dt, t_final = 0.0, 0.1, 150.0
    state = np.array([0.0, 0.0, 0.0, 0.0]) # x, y, vx, vy
    empuje = 15000.0 # Empuje constante
    
    # Almacenamos primero los datos 'limpios' de la simulación física
    datos_nominales = []
    while t <= t_final and state[1] >= 0:
        datos_nominales.append([t, state[0], state[1], state[2], state[3]])
        state = rk4_step(cinematica_cohete, t, state, dt, masa, empuje, angulo, cd)
        t += dt

    # Convertimos a array de NumPy para manipulación eficiente
    datos = np.array(datos_nominales)
    
    # --- Adición de Ruido Gaussiano (Sensores Comerciales) ---
    # Desviaciones estándar realistas:
    # Altitud (y): ~2.0 m (Fusión GPS/Barómetro)
    # Velocidades (vx, vy): ~0.1 m/s (IMU de grado comercial/Filtro de Kalman)
    sigma_y = 2.0
    sigma_v = 0.1
    
    # Aplicar ruido solo a las columnas solicitadas: y (2), vx (3), vy (4)
    # Mantenemos x (1) y t (0) según la instrucción específica
    datos[:, 2] += np.random.normal(0, sigma_y, size=len(datos))
    datos[:, 3] += np.random.normal(0, sigma_v, size=len(datos))
    datos[:, 4] += np.random.normal(0, sigma_v, size=len(datos))

    # Exportar a CSV
    with open(salida, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'vx', 'vy'])
        writer.writerows(datos)
    
    print(f"🚀 Telemetría con ruido instrumental generada en: {salida}")
    print(f"📊 Estadísticas: {len(datos)} muestras | σ_y={sigma_y}m | σ_v={sigma_v}m/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador de Telemetría Aeroespacial SciML con Ruido')
    parser.add_argument('--masa', type=float, default=500.0, help='Masa del cohete en kg')
    parser.add_argument('--cd', type=float, default=0.3, help='Coeficiente de arrastre verdadero')
    parser.add_argument('--angulo', type=float, default=85.0, help='Ángulo de lanzamiento inicial')
    parser.add_argument('--salida', type=str, default='data/vuelo.csv', help='Ruta del CSV')
    
    args = parser.parse_args()
    simular(args.masa, args.cd, args.angulo, args.salida)
