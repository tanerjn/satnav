

### Python Implementation (`propagate.py`)



import csv
import math
import matplotlib.pyplot as plt
import numpy as np

# Simulation Parameters
T_TOTAL_SEC = 5400  # Total simulation time (90 minutes / 1 orbit)
TIME_STEP = 60  # Time step in seconds
GM = 3.986004418e14  # Earth's gravitational constant (m^3/s^2)
R_EARTH = 6371000.0  # Earth radius in meters

# Simple circular low-earth orbit (LEO) parameters
altitude = 500000.0  # 500 km altitude
r_orbit = R_EARTH + altitude
n = math.sqrt(GM / (r_orbit**3))  # Mean motion

ephemeris_data = []
times, lats, lons, xs, ys, zs = [], [], [], [], [], []

print("Running orbital propagation and generating outputs...")

for t in range(0, T_TOTAL_SEC + 1, TIME_STEP):
  # Simple orbital mechanics calculations (Circular equatorial/inclined approximation)
  angle = n * t
  x = r_orbit * math.cos(angle)
  y = r_orbit * math.sin(angle)
  z = (
      r_orbit * 0.4 * math.sin(angle * 2)
  )  # Adding an orbital inclination effect

  # Convert to Lat/Lon/Alt ground track approximation
  r_mag = math.sqrt(x**2 + y**2 + z**2)
  lat = math.asin(z / r_mag) * (180.0 / math.pi)
  lon = (math.atan2(y, x) * (180.0 / math.pi) - (t * 0.041067)) % 360.0  # Earth rot
  if lon > 180:
    lon -= 360

  alt = r_mag - R_EARTH

  # Save for CSV
  ephemeris_data.append([t, x, y, z, lat, lon, alt])

  # Save for plotting
  times.append(t)
  xs.append(x / 1000.0)  # km
  ys.append(y / 1000.0)
  zs.append(z / 1000.0)
  lats.append(lat)
  lons.append(lon)

# 1. Generate ephemeris_output.csv
csv_filename = "ephemeris_output.csv"
with open(csv_filename, mode="w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(
      ["Time_s", "X_m", "Y_m", "Z_m", "Latitude_deg", "Longitude_deg", "Altitude_m"]
  )
  writer.writerows(ephemeris_data)
print(f"-> Successfully generated {csv_filename}")

# 2. Generate orbit_plot.png (3D Orbit Trajectory)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.plot(xs, ys, zs, label="Satellite Orbit", color="b", linewidth=2)
ax.set_title("3D Orbit Trajectory")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
plt.savefig("orbit_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("-> Successfully generated orbit_plot.png")

# 3. Generate ground_track.png (2D Ground Track)
plt.figure(figsize=(10, 5))
plt.plot(lons, lats, color="r", marker=".", linestyle="-", linewidth=1)
plt.title("Satellite Ground Track")
plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
plt.grid(True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.savefig("ground_track.png", dpi=300, bbox_inches="tight")
plt.close()
print("-> Successfully generated ground_track.png")

print("All tasks completed successfully!")
