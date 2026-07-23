import pandas as pd
import numpy as np
from pyproj import Geod

# =====================================================================
# STEP 0: Load data for a single test year
# =====================================================================
cost_data = pd.read_csv('data/processed/cost_data.csv')
years = [2014, 2015, 2016, 2017]
for yr in years:
    data = cost_data[cost_data['Year'] == yr]

    field_id = data['Field ID'].unique()
    N = len(field_id)
    year_vec = np.full(N, yr)

    months = ['April', 'May', 'June', 'July']
    month_suffix = {'April': 'apr', 'May': 'may', 'June': 'jun', 'July': 'jul'}


    # =====================================================================
    # STEP 1: Per-month outcome/covariate arrays
    # (unchanged logic, just a small helper to avoid 4x repetition)
    # =====================================================================
    def month_slice(data, month, col):
        return np.array(data.loc[data['Month'] == month, col])


    n = {month_suffix[m]: month_slice(data, m, 'Hill') for m in months}
    y = {month_suffix[m]: month_slice(data, m, 'w/PM') for m in months}
    a = {month_suffix[m]: month_slice(data, m, 'Area_Hectares') for m in months}
    s = {month_suffix[m]: month_slice(data, m, 'Sprays') for m in months}
    sI1 = {month_suffix[m]: month_slice(data, m, 'Initial Strain') for m in months}

    sI2 = np.ones(N)


    # =====================================================================
    # STEP 2: Cultivar susceptibility indicators (vectorized, no loop)
    # =====================================================================
    susceptibility = (
        data[data['Month'] == 'May']
        .drop_duplicates('Field ID')
        .set_index('Field ID')
        .loc[field_id, ['Susceptibility to V6 Strains', 'Susceptibility to non-V6 Strains']]
    )
    sus_r6 = susceptibility['Susceptibility to V6 Strains'].to_numpy()
    sus_nonr6 = susceptibility['Susceptibility to non-V6 Strains'].to_numpy()

    tI1 = ((sus_r6 == 1) & (sus_nonr6 == 0)).astype(float)
    tI2 = ((sus_r6 == 1) & (sus_nonr6 == 1)).astype(float)


    # =====================================================================
    # STEP 3: Wind run per direction, per source yard
    # =====================================================================
    def normalize_rows(array):
        row_sums = array.sum(axis=1, keepdims=True)
        return array / row_sums


    # NOTE ON UNITS: the source methodology specifies only "scale average
    # wind speed by the percentage of time for each cardinal direction" --
    # no unit conversion step is described. The original script's mph -> m/s
    # -> km/s conversion chain (0.44704 * 0.001) was not part of the
    # documented method and shrank wind_run by ~3 orders of magnitude,
    # which was the likely cause of gamma/alpha being unidentifiable.
    # wind_run is kept here in its raw reported units (mph), matching the
    # method as written. gamma's prior should be calibrated to this scale.
    wind_run = {}
    for m in months:
        suf = month_suffix[m]
        pct_time = np.array(data[data['Month'] == m].filter(like='Percent')) / 100
        pct_time = normalize_rows(pct_time)
        speed = np.array(data[data['Month'] == m].filter(like='Avg WS'))
        wind_run[suf] = speed * pct_time  # shape (N, 16), units: mph (raw, as reported)


    # =====================================================================
    # STEP 4: Angle helpers
    # =====================================================================
    def standard_form(angle):
        """Convert a compass bearing (radians, clockwise from north) to
        standard mathematical form (radians, counterclockwise from east)."""
        theta = np.pi / 2 - angle
        return np.mod(theta, 2 * np.pi)


    def wrap_to_pi(angle):
        """Wrap an angle (radians) to the interval (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi


    cardinal_directions = np.array([
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ])
    cardinal_angles = np.linspace(0, 2 * np.pi, 17)[:-1]
    cardinal_angles = standard_form(cardinal_angles)


    # =====================================================================
    # STEP 5: Geodesic distance + bearing between every yard pair
    # Vectorized in a single pyproj call instead of an N x N python loop.
    # =====================================================================
    coords = (
        data.drop_duplicates('Field ID')
        .set_index('Field ID')
        .loc[field_id, ['Centroid Lat', 'Centroid Long']]
    )
    lats = coords['Centroid Lat'].to_numpy()
    lons = coords['Centroid Long'].to_numpy()

    lat_i, lat_j = np.meshgrid(lats, lats, indexing='ij')
    lon_i, lon_j = np.meshgrid(lons, lons, indexing='ij')

    wgs84_geod = Geod(ellps='WGS84')
    az_ij, _, dist_m = wgs84_geod.inv(lon_i.ravel(), lat_i.ravel(),
                                    lon_j.ravel(), lat_j.ravel())

    distance = dist_m.reshape(N, N) * 0.001  # meters -> km
    bearing_tensor = standard_form(np.deg2rad(az_ij.reshape(N, N)))
    np.fill_diagonal(bearing_tensor, np.nan)  # undefined self-bearing


    # =====================================================================
    # STEP 6: Wind projection onto the bearing between each yard pair
    #
    # KEY SIMPLIFICATION: the original script builds a 2D wind vector
    # (wind_run * [cos(theta), sin(theta)]) and dot-products it against a
    # unit bearing vector [cos(bearing), sin(bearing)]. That's exactly:
    #
    #   dot = wind_run * (cos(theta)cos(bearing) + sin(theta)sin(bearing))
    #       = wind_run * cos(theta - bearing)
    #
    # So the entire vector construction + dot product + norm division
    # (norm is always 1, since it's built from cos/sin) collapses into a
    # single cosine term. No vectors, no transposes, no repeated blocks.
    # =====================================================================
    def project_wind(wind_run_month, bearing_tensor, cardinal_angles):
        """
        wind_run_month : (N, K) wind run per source yard per cardinal direction
        bearing_tensor : (N, N) bearing angle (standard form, radians), i -> j
        cardinal_angles: (K,) cardinal directions (standard form, radians)

        Returns (N, N) mean scalar projection of wind onto the i->j bearing,
        averaged over the cardinal directions within +/- 90 degrees of that
        bearing (the "downwind cone"), with self-pairs (i==j) set to 0.
        """
        diff = cardinal_angles[None, None, :] - bearing_tensor[:, :, None]
        diff = wrap_to_pi(diff)

        within_cone = np.abs(diff) < (np.pi / 2)          # (N, N, K)
        proj = wind_run_month[:, None, :] * np.cos(diff)  # (N, N, K)

        proj_masked = np.where(within_cone, proj, np.nan)
        with np.errstate(invalid='ignore'):
            result = np.nanmean(proj_masked, axis=-1)

        result = np.nan_to_num(result, nan=0.0)  # covers the i==j / undefined-bearing case
        np.fill_diagonal(result, 0.0)
        return result


    wind = {
        month_suffix[m]: project_wind(wind_run[month_suffix[m]], bearing_tensor, cardinal_angles)
        for m in months
    }


    # =====================================================================
    # STEP 7: Save
    # =====================================================================
    np.savez(
        f'data/processed/data_{yr}_test',
        field_id=field_id.flatten(), year_vec=year_vec, N=N, M=N,
        distance=distance, sI2=sI2, tI1=tI1, tI2=tI2,
        y_apr=y['apr'], n_apr=n['apr'], a_apr=a['apr'], wind_apr=wind['apr'], sI1_apr=sI1['apr'], s_apr=s['apr'],
        y_may=y['may'], n_may=n['may'], a_may=a['may'], wind_may=wind['may'], sI1_may=sI1['may'], s_may=s['may'],
        y_jun=y['jun'], n_jun=n['jun'], a_jun=a['jun'], wind_jun=wind['jun'], sI1_jun=sI1['jun'], s_jun=s['jun'],
        y_jul=y['jul'], n_jul=n['jul'], a_jul=a['jul'], wind_jul=wind['jul'], sI1_jul=sI1['jul'], s_jul=s['jul'],
    )

    print(f'Processed data for year {yr} saved.')