import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing file
data = pd.read_csv('../link/components/radiation_diagram/rad-sat-mon.csv')

phi = np.radians(data['Phi[deg]'])
theta = np.radians(data['Theta[deg]'])
gain = data['dB(GainTotal)']

# to retangular coordinates
r = 10 ** (gain / 10)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Plot 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=gain, cmap='viridis', marker='.')

# plot configurations
ax.set_title('Diagrama de Radiação 3D', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Ganho Total (dB)', fontsize=12)

plt.show()
