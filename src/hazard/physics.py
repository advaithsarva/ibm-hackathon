"""Governing physics (§3.2) — computes the inputs that feed the hazard formulas in §3.1.

Pure functions, no state, no I/O. Each returns SI units unless the name says otherwise.
Every constant a site would calibrate is an argument with a documented default, not a
literal buried in the body.

Standalone:
    python -m src.hazard.physics --demo
"""
import math

G_ACCEL = 9.80665  # m/s^2, for PGA in g <-> m/s^2


# --- 1. Floods -------------------------------------------------------------------

def manning_discharge(n, area_m2, hydraulic_radius_m, friction_slope):
    """Q = (1/n) * A * Rh^(2/3) * S^(1/2)  — channel discharge, m^3/s.

    n: channel roughness (0.03 natural stream, 0.013 concrete).
    Compare against bankfull capacity to flag overtopping.
    """
    if n <= 0:
        raise ValueError("Manning roughness n must be > 0")
    if friction_slope < 0:
        raise ValueError("friction slope must be >= 0")
    return (1.0 / n) * area_m2 * hydraulic_radius_m ** (2.0 / 3.0) * math.sqrt(friction_slope)


def flood_depth(stage_m, dem_elevation_m):
    """Depth(x,y) = H_stage - Z_DEM. Negative means dry; clamped to 0."""
    return max(0.0, stage_m - dem_elevation_m)


# --- 2. Cyclones -----------------------------------------------------------------

def holland_pressure(r_km, central_pressure_hpa, ambient_pressure_hpa, rmw_km, b=1.5):
    """P(r) = Pc + (Pn - Pc) * exp[-(Rm/r)^B]  — radial pressure profile, hPa.

    b: Holland shape parameter, typically 1.0-2.5. Calibrate per basin.
    """
    if r_km <= 0:
        raise ValueError("radius must be > 0")
    return central_pressure_hpa + (ambient_pressure_hpa - central_pressure_hpa) * math.exp(
        -((rmw_km / r_km) ** b)
    )


def holland_wind(r_km, central_pressure_hpa, ambient_pressure_hpa, rmw_km, b=1.5,
                 air_density=1.15):
    """Gradient wind from the same profile, m/s. Peaks at the radius of maximum wind."""
    if r_km <= 0:
        raise ValueError("radius must be > 0")
    dp_pa = (ambient_pressure_hpa - central_pressure_hpa) * 100.0
    scale = (rmw_km / r_km) ** b
    return math.sqrt(b * dp_pa * scale * math.exp(-scale) / air_density)


# --- 3. Droughts -----------------------------------------------------------------

def vci(ndvi, ndvi_min, ndvi_max):
    """VCI = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)  — percent.

    Below 35% is severe vegetation stress.
    """
    span = ndvi_max - ndvi_min
    if span <= 0:
        raise ValueError("ndvi_max must exceed ndvi_min")
    return 100.0 * (ndvi - ndvi_min) / span


# --- 4. Earthquakes --------------------------------------------------------------

def gmpe_pga(magnitude_mw, epicentral_distance_km, focal_depth_km,
             c1=-1.72, c2=0.98, c3=1.30):
    """ln(PGA) = c1 + c2*M - c3*ln(sqrt(R^2 + h^2))  — returns PGA in m/s^2.

    c1..c3 are regional attenuation coefficients; the defaults are a generic shallow-crust
    set and MUST be refit per region before anyone trusts an absolute number.
    """
    r = math.sqrt(epicentral_distance_km ** 2 + focal_depth_km ** 2)
    if r <= 0:
        raise ValueError("hypocentral distance must be > 0")
    return math.exp(c1 + c2 * magnitude_mw - c3 * math.log(r))


def amplify_vs30(pga_ms2, vs30_ms, vs30_reference=760.0, exponent=-0.36):
    """Soft-soil site amplification. Low vs30 amplifies shaking.

    exponent is the site-response slope; -0.36 is a common empirical value.
    """
    if vs30_ms <= 0:
        raise ValueError("vs30 must be > 0")
    return pga_ms2 * (vs30_ms / vs30_reference) ** exponent


# --- 5. Landslides ---------------------------------------------------------------

def factor_of_safety(cohesion_kpa, unit_weight_kn_m3, depth_m, slope_deg,
                     friction_angle_deg, water_table_height_m=0.0,
                     water_unit_weight_kn_m3=9.81):
    """FS = [c' + (gz - gw*hw) cos^2(t) tan(phi')] / [gz sin(t) cos(t)]

    FS > 1 stable, FS <= 1 imminent failure. Dimensionless.
    """
    t = math.radians(slope_deg)
    if math.sin(t) == 0:
        return float("inf")  # flat ground never fails by this mechanism
    gz = unit_weight_kn_m3 * depth_m
    effective = gz - water_unit_weight_kn_m3 * water_table_height_m
    numerator = cohesion_kpa + effective * math.cos(t) ** 2 * math.tan(math.radians(friction_angle_deg))
    return numerator / (gz * math.sin(t) * math.cos(t))


# --- 6. Lightning ----------------------------------------------------------------

def flash_rate(cape_j_kg, updraft_velocity_ms, a=1.0):
    """F_L = a * CAPE^(1/2) * w_max^2  — relative flash rate.

    a is an unnormalized site coefficient; the output is only meaningful as a ratio
    until it is calibrated against an observed strike network.
    """
    if cape_j_kg < 0:
        raise ValueError("CAPE must be >= 0")
    return a * math.sqrt(cape_j_kg) * updraft_velocity_ms ** 2


def _demo():
    print(f"manning Q          = {manning_discharge(0.03, 120.0, 2.4, 0.001):8.2f} m3/s")
    print(f"holland P(30km)    = {holland_pressure(30, 950, 1010, 25):8.2f} hPa")
    print(f"holland V(25km)    = {holland_wind(25, 950, 1010, 25):8.2f} m/s")
    print(f"vci                = {vci(0.42, 0.18, 0.76):8.2f} %")
    pga = gmpe_pga(6.8, 12.0, 10.0)
    print(f"gmpe PGA           = {pga:8.3f} m/s2  ({pga / G_ACCEL:.3f} g)")
    print(f"  amplified vs30=280= {amplify_vs30(pga, 280):8.3f} m/s2")
    print(f"factor of safety   = {factor_of_safety(5.0, 18.0, 2.0, 35.0, 30.0, 0.5):8.3f}")
    print(f"flash rate         = {flash_rate(2800, 18.0):8.2f}")


if __name__ == "__main__":
    _demo()
