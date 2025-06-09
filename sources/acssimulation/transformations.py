import numpy as np
from sgp4.api import jday

def jd_to_gmst(jd):
    """Calcula o Tempo Sidério de Greenwich (GMST) em radianos."""
    t = (jd - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    gmst_hours = (gmst_sec % 86400) / 3600
    return gmst_hours * 15 * np.pi / 180  # Converte para radianos

def ned_to_ecef(lat, lon, v_ned):
    """Converte vetor do sistema NED para ECEF."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    R = np.array([
        [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lon_rad), -np.cos(lat_rad)*np.cos(lon_rad)],
        [-np.sin(lat_rad)*np.sin(lon_rad),  np.cos(lon_rad), -np.cos(lat_rad)*np.sin(lon_rad)],
        [ np.cos(lat_rad),                  0,              -np.sin(lat_rad)]
    ])
    return R.T @ v_ned

def ecef_to_eci(v_ecef, jd):
    """Converte vetor do sistema ECEF para ECI."""
    theta = jd_to_gmst(jd)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R @ v_ecef

def quat_to_rot_matrix(q):
    """Converte quaternion para matriz de rotação."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)]
    ])