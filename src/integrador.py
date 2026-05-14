import numpy as np
from typing import Callable, Any

def rk4_step(
    f: Callable[[float, np.ndarray, Any], np.ndarray], 
    t: float, 
    y: np.ndarray, 
    dt: float, 
    *args: Any
) -> np.ndarray:
    """
    Realiza un paso de integración numérica utilizando el método de Runge-Kutta de 4º orden (RK4).

    Args:
        f: Función que calcula la derivada del estado dy/dt. Debe aceptar (t, y, *args).
        t: Tiempo actual en la simulación.
        y: Vector de estado actual [x, y, vx, vy].
        dt: Paso de tiempo (paso de integración).
        *args: Argumentos adicionales pasados a la función de dinámica (masa, empuje, etc.).

    Returns:
        np.ndarray: Vector de estado en el tiempo t + dt.
    """
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt/2 * k1, *args)
    k3 = f(t + dt/2, y + dt/2 * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def cinematica_cohete(
    t: float, 
    state: np.ndarray, 
    masa: float, 
    empuje: float, 
    angulo_deg: float, 
    cd: float
) -> np.ndarray:
    """
    Define el sistema de Ecuaciones Diferenciales Ordinarias (EDOs) para la balística del cohete.

    Calcula las derivadas temporales del estado [x, y, vx, vy] considerando gravedad,
    empuje variable por ángulo y arrastre aerodinámico con modelo atmosférico exponencial.

    Args:
        t: Tiempo actual (s).
        state: Vector de estado [x, y, vx, vy].
        masa: Masa total del vehículo (kg).
        empuje: Magnitud del empuje del motor (N).
        angulo_deg: Ángulo de lanzamiento inicial (grados).
        cd: Coeficiente de arrastre (adimensional).

    Returns:
        np.ndarray: Derivadas [dx/dt, dy/dt, dvx/dt, dvy/dt].
    """
    x, y_alt, vx, vy = state
    
    # Constantes físicas
    G: float = 9.81         # Aceleración gravitacional (m/s^2)
    RHO_0: float = 1.225    # Densidad del aire al nivel del mar (kg/m^3)
    H: float = 8500.0       # Altura de escala atmosférica (m)
    AREA: float = 3.14      # Área transversal de referencia (m^2)
    
    # Perfil de vuelo: Gravity Turn simplificado
    # Mantiene el ángulo inicial 20s y luego rota 2 grados por segundo
    if t < 20.0:
        theta = np.radians(angulo_deg)
    else:
        theta = np.radians(max(0.0, angulo_deg - (t - 20.0) * 2.0))
    
    # Modelo Atmosférico (Densidad variable con la altitud)
    # Se trunca a los 100km (Karman Line aproximada)
    rho = RHO_0 * np.exp(-max(0.0, y_alt) / H) if y_alt < 100000.0 else 0.0
    
    # Dinámica de Arrastre (Drag Force)
    v_mag = np.sqrt(vx**2 + vy**2) + 1e-8 # Regularización para evitar división por cero
    f_drag = 0.5 * rho * v_mag**2 * cd * AREA
    
    # Ecuaciones de Movimiento (2ª Ley de Newton)
    # Componentes de aceleración: (Empuje - Arrastre - Gravedad) / Masa
    dx = vx
    dy = vy
    dvx = (empuje * np.cos(theta) - f_drag * (vx / v_mag)) / masa
    dvy = (empuje * np.sin(theta) - f_drag * (vy / v_mag)) / masa - G
    
    return np.array([dx, dy, dvx, dvy])
