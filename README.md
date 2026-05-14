# Inverse Drag Discovery: Physics-Informed Neural Networks for Parameter Identification

Este repositorio contiene una implementación de **Physics-Informed Neural Networks (PINNs)** para resolver un problema inverso en la dinámica de vuelo suborbital: la identificación del coeficiente de arrastre ($C_d$) y la masa ($m$) de un vehículo a partir de telemetría ruidosa.

## 1. Resumen del Problema Inverso

En la ingeniería aeroespacial, determinar las características aerodinámicas reales a partir de datos de vuelo es crítico. Este proyecto aborda la recuperación de parámetros físicos ocultos mediante la integración de leyes físicas fundamentales en el proceso de optimización de una red neuronal. A diferencia de un enfoque de regresión tradicional, la PINN actúa como un regularizador basado en la física, permitiendo descubrir constantes como el $C_d$ incluso cuando los datos observados presentan ruido instrumental significativo.

## 2. Dinámica del Sistema y Ecuaciones Diferenciales

El movimiento del vehículo se modela mediante un sistema de Ecuaciones Diferenciales Ordinarias (EDOs) en 2D, considerando un empuje constante, gravedad y arrastre aerodinámico en una atmósfera exponencial.

Sea el vector de estado $\mathbf{s} = [x, y, v_x, v_y]^T$, la dinámica está gobernada por:

$$
\begin{aligned}
\frac{dx}{dt} &= v_x \\
\frac{dy}{dt} &= v_y \\
\frac{dv_x}{dt} &= \frac{1}{m} \left( T \cos(\theta) - F_d \frac{v_x}{\|\mathbf{v}\|} \right) \\
\frac{dv_y}{dt} &= \frac{1}{m} \left( T \sin(\theta) - F_d \frac{v_y}{\|\mathbf{v}\|} \right) - g
\end{aligned}
$$

Donde la fuerza de arrastre ($F_d$) y la densidad atmosférica ($\rho$) se definen como:

$$
F_d = \frac{1}{2} \rho(y) \|\mathbf{v}\|^2 C_d A, \quad \rho(y) = \rho_0 e^{-y/H}
$$

- $T$: Empuje (Thrust).
- $\theta$: Ángulo de vuelo (Gravity turn perfilado).
- $A$: Área de referencia.
- $\rho_0$: Densidad al nivel del mar.
- $H$: Altura de escala.

## 3. Arquitectura de la PINN

La arquitectura consiste en un Perceptrón Multicapa (MLP) que aproxima la solución del sistema dinámico $t \to \hat{\mathbf{s}}(t)$.

- **Arquitectura de la Red**: 1 entrada ($t$), 2 capas ocultas de 64 neuronas con activación `Tanh`, y 4 salidas ($x, y, v_x, v_y$).
- **Parámetros Aprendibles**: Además de los pesos de la red, el modelo optimiza directamente los parámetros físicos $\theta_{phys} = \{C_d, m\}$.
- **Función de Pérdida Multiobjetivo**:
  $$\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$$
  Donde $\mathcal{L}_{data}$ es el error cuadrático medio respecto a la telemetría y $\mathcal{L}_{physics}$ es el residuo de las EDOs evaluado mediante diferenciación automática (Autograd).

## 4. Estructura del Proyecto

- `src/simulador.py`: Generador de telemetría sintética con ruido gaussiano utilizando integración RK4.
- `src/integrador.py`: Implementación del método de Runge-Kutta de 4º orden y definición de las EDOs.
- `src/pinn_cd.py`: Core del proyecto. Implementación de la PINN en PyTorch, bucle de entrenamiento y visualización.
- `cli/generar_reporte.sh`: Pipeline MLOps para simulación y auditoría de datos.

## 5. Instrucciones de Ejecución

### Pre-requisitos
Asegúrese de tener un entorno virtual configurado e instalar las dependencias:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Flujo de Trabajo
1. **Generar Datos de Vuelo**:
   ```bash
   python3 src/simulador.py --masa 500 --cd 0.3 --salida data/vuelo.csv
   ```

2. **Entrenar PINN y Descubrir Parámetros**:
   ```bash
   python3 src/pinn_cd.py
   ```
   Durante el entrenamiento, podrá observar la convergencia de $C_d$ y la masa en tiempo real. Al finalizar, se generará una animación en `docs/convergencia.gif` y una comparativa estática en `docs/comparativa_pinn.png`.

3. **Ejecutar Pipeline Completo**:
   ```bash
   bash cli/generar_reporte.sh
   ```

---
**Investigación SciML** | Desarrollado para el descubrimiento automatizado de leyes físicas.
