import numpy as np

def simulate_attitude(I, omega0, q0, M, dt, T):
    """
    Simula a evolução da atitude de um corpo rígido usando quaternions com integração RK4.

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

    # Funções auxiliares para cálculo das derivadas
    def domega_dt(omega_vec):
        ωx, ωy, ωz = omega_vec
        dx = (M[0] - (ωy * ωz * (Izz - Iyy))) / Ixx
        dy = (M[1] - (ωz * ωx * (Ixx - Izz))) / Iyy
        dz = (M[2] - (ωx * ωy * (Iyy - Ixx))) / Izz
        return np.array([dx, dy, dz])
    
    def dq_dt(omega_vec, q_vec):
        ωx, ωy, ωz = omega_vec
        Omega_prime = np.array([
            [ 0,    ωz,  -ωy,  ωx],
            [-ωz,   0,   ωx,  ωy],
            [ ωy, -ωx,   0,   ωz],
            [-ωx, -ωy, -ωz,   0 ]
        ])
        return 0.5 * Omega_prime @ q_vec

    for i in range(steps):
        t = i * dt
        time_hist.append(t)
        omega_hist.append(omega.copy())
        quat_hist.append(q.copy())

        # Integração RK4 para omega
        k1_omega = domega_dt(omega)
        k2_omega = domega_dt(omega + 0.5 * dt * k1_omega)
        k3_omega = domega_dt(omega + 0.5 * dt * k2_omega)
        k4_omega = domega_dt(omega + dt * k3_omega)
        omega += (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

        # Integração RK4 para quaternion
        k1_q = dq_dt(omega, q)
        k2_q = dq_dt(omega, q + 0.5 * dt * k1_q)
        k3_q = dq_dt(omega, q + 0.5 * dt * k2_q)
        k4_q = dq_dt(omega, q + dt * k3_q)
        q += (dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        
        # Normalização do quaternion
        q /= np.linalg.norm(q)

    return np.array(time_hist), np.array(omega_hist), np.array(quat_hist)
