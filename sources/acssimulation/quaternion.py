import numpy as np

def simulate_attitude(I, omega0, q0, M, dt, T):
    """
    Simula a evolução da atitude de um corpo rígido usando quaternions.

    Parâmetros:
    - I: array-like, [Ixx, Iyy, Izz] (kg*m²)
    - omega0: array-like, [ωx, ωy, ωz] inicial (rad/s)
    - q0: array-like, quaternion inicial [q0, q1, q2, q3]
    - M: array-like, torques aplicados [Mx, My, Mz]
    - dt: float, passo de tempo (s)
    - T: float, duração total (s)

    Retorna:
    - time_hist: array com os instantes de tempo
    - omega_hist: histórico das velocidades angulares
    - quat_hist: histórico dos quaternions
    """

    Ixx, Iyy, Izz = I
    omega = np.array(omega0, dtype=float)
    q = np.array(q0, dtype=float)
    steps = int(T / dt)

    omega_hist = []
    quat_hist = []
    time_hist = []

    for i in range(steps):
        t = i * dt
        ωx, ωy, ωz = omega

        # Equações diferenciais da velocidade angular (Eq. 4.44–4.46)
        domega_x = (M[0] - (ωy * ωz * (Izz - Iyy))) / Ixx
        domega_y = (M[1] - (ωz * ωx * (Ixx - Izz))) / Iyy
        domega_z = (M[2] - (ωx * ωy * (Iyy - Ixx))) / Izz
        domega = np.array([domega_x, domega_y, domega_z])

        # Atualiza omega
        omega += domega * dt

        # Matriz Ω′ (Eq. 4.48)
        Omega_prime = np.array([
            [ 0,    ωz,  -ωy,  ωx],
            [-ωz,   0,   ωx,  ωy],
            [ ωy, -ωx,   0,   ωz],
            [-ωx, -ωy, -ωz,   0 ]
        ])

        # Derivada do quaternion (Eq. 4.47)
        dq = 0.5 * Omega_prime @ q

        # Atualiza quaternion
        q += dq * dt
        q /= np.linalg.norm(q)  # normaliza

        # Armazena histórico
        time_hist.append(t)
        omega_hist.append(omega.copy())
        quat_hist.append(q.copy())

    return np.array(time_hist), np.array(omega_hist), np.array(quat_hist)
