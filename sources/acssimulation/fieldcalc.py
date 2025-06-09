from sgp4.api import Satrec, jday
import numpy as np
import matplotlib.pyplot as plt
import wmm2020
import datetime

def field_calc(s, t, num_points, start_date):
    # Inicializa satélite
    satellite = Satrec.twoline2rv(s, t)

    delta_t = 90 * 60 / num_points  # passo em segundos (~90min)

    # Listas para resultados
    times = []
    east_vals = []
    north_vals = []
    down_vals = []
    lats = []
    lons = []
    alts = []

    # Simulação da órbita
    for i in range(num_points):
        current_time = start_date + datetime.timedelta(seconds=i * delta_t)
        jd, fr = jday(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second + current_time.microsecond / 1e6)

        e, r, v = satellite.sgp4(jd, fr)

        if e == 0:
            x, y, z = r
            norm_r = np.linalg.norm(r)
            lat = np.degrees(np.arcsin(z / norm_r))
            lon = np.degrees(np.arctan2(y, x))
            alt_km = norm_r - 6371  # raio médio da Terra em km

            # Ano decimal
            year = current_time.year
            day_of_year = current_time.timetuple().tm_yday + (current_time.hour + current_time.minute/60 + current_time.second/3600)/24
            yeardec = year + (day_of_year / 365.25)

            # Campo magnético
            mag = wmm2020.wmm(lat, lon, alt_km, yeardec)

            # Converte nT para Gauss (1 nT = 1e-5 Gauss)
            east_vals.append(mag["east"].values.item() * 1e-5)
            north_vals.append(mag["north"].values.item() * 1e-5)
            down_vals.append(mag["down"].values.item() * 1e-5)
            times.append(current_time)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt_km)
        else:
            print(f"Erro na propagação SGP4 em {current_time}: código {e}")

    return times, east_vals, north_vals, down_vals, lats, lons, alts