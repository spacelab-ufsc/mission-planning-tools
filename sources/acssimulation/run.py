import numpy as np
import matplotlib.pyplot as plt
from quaternion import simulate_attitude
from fieldcalc import field_calc
import datetime
from scipy.interpolate import interp1d
import transformations

# TLE do satélite
s = '1 41866U 16071A   25118.86953697 -.00000191  00000+0  00000+0 0  9990'
t = '2 41866   0.0503 276.7676 0000706 197.1672 331.6443  1.00128628 30940'

# Parâmetros da simulação
num_points = 100  # quantidade de pontos na órbita
start_date = datetime.datetime(2019, 12, 20, 0, 0, 0)

# ===== 1. Simulação do Campo Magnético =====
times, east, north, down, lats, lons, alts = field_calc(s, t, num_points, start_date)

# Plot dos componentes originais (opcional)
plt.figure(figsize=(12, 6))
plt.plot(times, east, label='East (G)', marker='o')
plt.plot(times, north, label='North (G)', marker='s')
plt.plot(times, down, label='Down (G)', marker='^')
plt.title('Componentes do Campo Magnético (WMM2020)')
plt.xlabel('Tempo')
plt.ylabel('Intensidade (Gauss)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===== 2. Conversão para Sistema ECI =====
# Calcular JD para cada tempo
jds = []
for t in times:
    jd, fr = transformations.jday(t.year, t.month, t.day, 
                                 t.hour, t.minute, t.second + t.microsecond/1e6)
    jds.append(jd + fr)

# Converter campo magnético de NED para ECI
mag_eci = []
for i in range(len(times)):
    # Monta vetor NED [North, East, Down]
    v_ned = np.array([north[i], east[i], down[i]])
    
    # Converte NED → ECEF → ECI
    v_ecef = transformations.ned_to_ecef(lats[i], lons[i], v_ned)
    v_eci = transformations.ecef_to_eci(v_ecef, jds[i])
    mag_eci.append(v_eci)

mag_eci = np.array(mag_eci)

# ===== 3. Simulação de Atitude =====
# Configurações do usuário
I = [0.0185, 0.0183, 0.0043]         # Momentos de inércia [kg*m²]
omega0 = [10, 6, 4]                   # Velocidade angular inicial [rad/s]
q0 = [0.5490, 0.8306, 0.0780, 0.0515] # Quaternion inicial
M = [0.0, 0.0, 0.0]                   # Torque nulo
dt = 0.1                              # Passo de tempo [s]
T = 5400                              # Duração total [s] (1.5 horas)

# Executa simulação de atitude
time_hist, omega_hist, quat_hist = simulate_attitude(I, omega0, q0, M, dt, T)

# Plot dos quaternions (opcional)
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

# ===== 4. Processamento Conjunto =====
# Converter tempos orbitais para segundos
t0 = start_date
time_sec = [(t - t0).total_seconds() for t in times]

# Interpolar campo magnético ECI para os tempos da simulação de atitude
interp_x = interp1d(time_sec, mag_eci[:, 0], kind='linear', fill_value="extrapolate")
interp_y = interp1d(time_sec, mag_eci[:, 1], kind='linear', fill_value="extrapolate")
interp_z = interp1d(time_sec, mag_eci[:, 2], kind='linear', fill_value="extrapolate")

# Obter valores interpolados
mag_x = interp_x(time_hist)
mag_y = interp_y(time_hist)
mag_z = interp_z(time_hist)

# Rotacionar para sistema de corpo usando quaternions
mag_body = []
for i in range(len(time_hist)):
    # Obter matriz de rotação do quaternion
    R = transformations.quat_to_rot_matrix(quat_hist[i])
    
    # Vetor campo magnético em ECI
    v_eci = np.array([mag_x[i], mag_y[i], mag_z[i]])
    
    # Rotacionar para sistema de corpo
    v_body = R @ v_eci
    mag_body.append(v_body)

mag_body = np.array(mag_body)

# ===== 5. Visualização dos Resultados =====
plt.figure(figsize=(14, 10))

# Gráfico do campo magnético em ECI
plt.subplot(2, 1, 1)
plt.plot(time_hist, mag_x, 'r--', label='ECI X')
plt.plot(time_hist, mag_y, 'g--', label='ECI Y')
plt.plot(time_hist, mag_z, 'b--', label='ECI Z')
plt.title('Campo Magnético no Sistema Inercial (ECI)')
plt.ylabel('Intensidade (Gauss)')
plt.legend()
plt.grid(True)

# Gráfico do campo magnético em sistema de corpo
plt.subplot(2, 1, 2)
plt.plot(time_hist, mag_body[:, 0], 'r-', label='Body X')
plt.plot(time_hist, mag_body[:, 1], 'g-', label='Body Y')
plt.plot(time_hist, mag_body[:, 2], 'b-', label='Body Z')
plt.title('Campo Magnético no Sistema de Corpo do Satélite')
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade (Gauss)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Gráfico 3D da trajetória do campo magnético no sistema de corpo
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(mag_body[:, 0], mag_body[:, 1], mag_body[:, 2])
ax.set_xlabel('Eixo X (Gauss)')
ax.set_ylabel('Eixo Y (Gauss)')
ax.set_zlabel('Eixo Z (Gauss)')
ax.set_title('Trajetória do Campo Magnético no Sistema de Corpo')
plt.tight_layout()
plt.show()

# ===== 6. Análise Complementar =====
# Calcular magnitude total do campo magnético
mag_total = np.linalg.norm(mag_body, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(time_hist, mag_total, 'm-', linewidth=2)
plt.title('Magnitude Total do Campo Magnético')
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade (Gauss)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcular variações por eixo
variation_x = np.max(mag_body[:, 0]) - np.min(mag_body[:, 0])
variation_y = np.max(mag_body[:, 1]) - np.min(mag_body[:, 1])
variation_z = np.max(mag_body[:, 2]) - np.min(mag_body[:, 2])

print("\nAnálise do Campo Magnético no Sistema de Corpo:")
print(f"Variação no eixo X: {variation_x:.6f} Gauss")
print(f"Variação no eixo Y: {variation_y:.6f} Gauss")
print(f"Variação no eixo Z: {variation_z:.6f} Gauss")
print(f"Média da magnitude: {np.mean(mag_total):.6f} Gauss")
print(f"Variação total: {np.max(mag_total) - np.min(mag_total):.6f} Gauss")

