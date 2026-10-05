import numpy as np

class LEOOrbitalPropagator:
    """
    Keplerian Low-Earth Orbit (LEO) Satellite Propagator.
    Configured by default for a 98° Sun-Synchronous Orbit (SSO) at 500 km altitude.
    Calculates sub-satellite coordinates (Lat, Lon, Alt), sun vector, eclipse shadow, and solar flux.
    """
    def __init__(self, altitude_km: float = 500.0, inclination_deg: float = 98.0):
        self.r_earth_km = 6371.0  # Earth mean radius
        self.mu_earth = 398600.4418  # Standard gravitational parameter (km^3/s^2)
        self.omega_earth = 7.2921159e-5  # Earth rotation rate (rad/s)
        self.solar_constant = 1361.0  # W/m^2 nominal solar flux at 1 AU

        self.altitude_km = altitude_km
        self.a = self.r_earth_km + altitude_km
        self.inc_rad = np.radians(inclination_deg)
        self.mean_motion = np.sqrt(self.mu_earth / (self.a ** 3)) # rad/s
        self.period_seconds = (2 * np.pi) / self.mean_motion # ~94 minutes

    def get_orbit_trajectory(self, num_points: int = 200) -> dict:
        """
        Generates 3D orbit trajectory path over 1 full orbit period.
        Returns dict with 3D Cartesian coordinates (X, Y, Z in km) and (Lat, Lon) series.
        """
        times = np.linspace(0, self.period_seconds, num_points)
        lats, lons, alt_list = [], [], []
        x_list, y_list, z_list = [], [], []

        for t in times:
            pos = self.propagate(t)
            lats.append(pos["lat"])
            lons.append(pos["lon"])
            alt_list.append(pos["alt"])
            x_list.append(pos["x_eci"])
            y_list.append(pos["y_eci"])
            z_list.append(pos["z_eci"])

        return {
            "lats": lats,
            "lons": lons,
            "altitudes": alt_list,
            "x_eci": x_list,
            "y_eci": y_list,
            "z_eci": z_list
        }

    def propagate(self, elapsed_seconds: float) -> dict:
        """
        Calculates satellite state at elapsed time t (seconds).
        """
        # True anomaly / argument of latitude u
        u = self.mean_motion * elapsed_seconds

        # Orbit plane Cartesian ECI coordinates
        x_orb = self.a * np.cos(u)
        y_orb = self.a * np.sin(u)

        # Rotate by inclination i
        x_eci = x_orb
        y_eci = y_orb * np.cos(self.inc_rad)
        z_eci = y_orb * np.sin(self.inc_rad)

        # Convert to ECEF spherical coordinates (Lat, Lon) accounting for Earth rotation
        r = np.sqrt(x_eci**2 + y_eci**2 + z_eci**2)
        lat_rad = np.arcsin(z_eci / r)
        lon_rad = np.arctan2(y_eci, x_eci) - self.omega_earth * elapsed_seconds

        # Normalize longitude to [-180, 180]
        lon_deg = np.degrees(lon_rad)
        lon_deg = (lon_deg + 180.0) % 360.0 - 180.0
        lat_deg = np.degrees(lat_rad)

        # Sun vector calculation (simplified solar ecliptic position)
        day_of_year = 80.0  # Approx vernal equinox
        sun_long_rad = np.radians((360.0 / 365.25) * day_of_year)
        sun_vec = np.array([np.cos(sun_long_rad), np.sin(sun_long_rad) * np.cos(np.radians(23.44)), np.sin(sun_long_rad) * np.sin(np.radians(23.44))])
        sun_vec = sun_vec / np.linalg.norm(sun_vec)

        # Satellite position vector normalized
        sat_vec = np.array([x_eci, y_eci, z_eci]) / r

        # Solar zenith angle dot product
        cos_zenith = np.dot(sat_vec, sun_vec)

        # Eclipse Shadow Determination (Cylindrical Earth shadow model)
        proj_length = np.dot(np.array([x_eci, y_eci, z_eci]), sun_vec)
        perp_dist = np.linalg.norm(np.array([x_eci, y_eci, z_eci]) - proj_length * sun_vec)

        is_eclipse = (proj_length < 0) and (perp_dist < self.r_earth_km)
        eclipse_status = "IN ECLIPSE (UMBRA)" if is_eclipse else "SUNLIT (DAYLIGHT)"

        # Instantaneous Solar Radiation Flux
        if is_eclipse:
            solar_flux = 0.0
        else:
            solar_flux = self.solar_constant * max(0.0, cos_zenith)

        return {
            "elapsed_seconds": elapsed_seconds,
            "lat": lat_deg,
            "lon": lon_deg,
            "alt": self.altitude_km,
            "x_eci": x_eci,
            "y_eci": y_eci,
            "z_eci": z_eci,
            "sun_vec": sun_vec.tolist(),
            "is_eclipse": is_eclipse,
            "eclipse_status": eclipse_status,
            "solar_flux": float(solar_flux)
        }
