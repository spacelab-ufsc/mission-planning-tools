import numpy as np
import matplotlib.pyplot as plt
from quaternion import simulate_attitude
from fieldcalc import field_calc
import datetime

# TLE do satélite
s = '1 41866U 16071A   25118.86953697 -.00000191  00000+0  00000+0 0  9990'
t = '2 41866   0.0503 276.7676 0000706 197.1672 331.6443  1.00128628 30940'
# Parâmetros da simulação
num_points = 100  # quantidade de pontos na órbita
start_date = datetime.datetime(2019, 12, 20, 0, 0, 0)

times, east_vals, north_vals, down_vals = field_calc(s, t, num_points, start_date)

# Plot dos componentes
plt.figure(figsize=(12, 6))
plt.plot(times, east_vals, label='East (G)', marker='o')
plt.plot(times, north_vals, label='North (G)', marker='s')
plt.plot(times, down_vals, label='Down (G)', marker='^')
plt.title('Componentes do Campo Magnético (WMM2020)')
plt.xlabel('Tempo')
plt.ylabel('Intensidade (Gauss)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()











# ===== Configurações do usuário =====
I = [10.0, 15.0, 20.0]  # Ixx, Iyy, Izz
omega0 = [0.1, 0.2, 0.3]  # rad/s
q0 = [1.0, 0.0, 0.0, 0.0]  # quaternion unitário inicial
M = [0.0, 0.0, 0.0]  # torque nulo
dt = 0.1
T = 100  # segundos

# ===== Executa simulação =====
time_hist, omega_hist, quat_hist = simulate_attitude(I, omega0, q0, M, dt, T)

# ===== Plot dos quaternions =====
plt.figure(figsize=(10, 6))
plt.plot(time_hist, quat_hist[:, 0], label='q0', color='blue')
plt.plot(time_hist, quat_hist[:, 1], label='q1', color='green')
plt.plot(time_hist, quat_hist[:, 2], label='q2', color='red')
plt.plot(time_hist, quat_hist[:, 3], label='q3', color='purple')

plt.xlabel('Tempo (s)')
plt.ylabel('Quaternions')
plt.title('Evolução da Atitude (Quaternion)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



