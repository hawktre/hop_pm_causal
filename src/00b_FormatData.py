import pandas as pd
import numpy as np
from pyproj import Geod

# Import cost data
cost_data = pd.read_csv('data/processed/cost_data.csv')

# Define year
yr = [2014, 2015, 2016, 2017]

for yr in yr:

    # Data for given year
    data = cost_data[cost_data['Year'] == yr]

    # Number of yards i (source)
    N = data['Field ID'].unique().shape[0]

    # Number of yards j (target)
    M = data['Field ID'].unique().shape[0]

    periods = [0, 1]

    # Create arrays containing the number of plants sampled in each yard
    n_apr = np.array(data.loc[data['Month'] == 'April', 'Hill'])
    n_may = np.array(data.loc[data['Month'] == 'May', 'Hill'])
    n_jun = np.array(data.loc[data['Month'] == 'June', 'Hill'])
    n_jul = np.array(data.loc[data['Month'] == 'July', 'Hill'])

    # Create arrays containing the number of diseased plants in each yard
    y_apr = np.array(data.loc[data['Month'] == 'April', 'w/PM'])
    y_may = np.array(data.loc[data['Month'] == 'May', 'w/PM'])
    y_jun = np.array(data.loc[data['Month'] == 'June', 'w/PM'])
    y_jul = np.array(data.loc[data['Month'] == 'July', 'w/PM'])

    # Create arrays containing the number of diseased plants in each OTHER yard
    z_may = np.array(data.loc[data['Month'] == 'May', 'w/PM'])
    z_jun = np.array(data.loc[data['Month'] == 'June', 'w/PM'])
    z_jul = np.array(data.loc[data['Month'] == 'July', 'w/PM'])

    # Create arrays containing the area (hectares) of each yard
    a_apr = np.array(data.loc[data['Month'] == 'April', 'Area_Hectares'])
    a_may = np.array(data.loc[data['Month'] == 'May', 'Area_Hectares'])
    a_jun = np.array(data.loc[data['Month'] == 'June', 'Area_Hectares'])
    a_jul = np.array(data.loc[data['Month'] == 'July', 'Area_Hectares'])

    # Create arrays containing the fungicide spray amount
    s_apr = np.array(data.loc[data['Month'] == 'April', 'Sprays'])
    s_may = np.array(data.loc[data['Month'] == 'May', 'Sprays'])
    s_jun = np.array(data.loc[data['Month'] == 'June', 'Sprays'])
    s_jul = np.array(data.loc[data['Month'] == 'July', 'Sprays'])

    # Create arrays for wind speed and percent time
    
    # Function to normalize the values so that they sum up to 1 for each row
    def normalize(array):
        row_sums = array.sum(axis=1, keepdims=True)
        return array / row_sums

    # April
    wind_percent_time_apr = np.array(data[data['Month'] == 'April'].filter(like='Percent')) / 100
    wind_percent_time_apr = normalize(wind_percent_time_apr)

    # May
    wind_percent_time_may = np.array(data[data['Month'] == 'May'].filter(like='Percent')) / 100
    wind_percent_time_may = normalize(wind_percent_time_may)

    # June
    wind_percent_time_jun = np.array(data[data['Month'] == 'June'].filter(like='Percent')) / 100
    wind_percent_time_jun = normalize(wind_percent_time_jun)

    # July
    wind_percent_time_jul = np.array(data[data['Month'] == 'July'].filter(like='Percent')) / 100
    wind_percent_time_jul = normalize(wind_percent_time_jul)

    # Select columns that start with 'Avg WS' for each month
    wind_speed_apr = np.array(data[data['Month'] == 'April'].filter(like='Avg WS'))
    wind_speed_may = np.array(data[data['Month'] == 'May'].filter(like='Avg WS'))
    wind_speed_jun = np.array(data[data['Month'] == 'June'].filter(like='Avg WS'))
    wind_speed_jul = np.array(data[data['Month'] == 'July'].filter(like='Avg WS'))
    

    # Compute wind run and convert from mi/hr to km/s
    wind_run_apr = wind_speed_apr * wind_percent_time_apr * 0.44704 * 0.001
    wind_run_may = wind_speed_may * wind_percent_time_may * 0.44704 * 0.001
    wind_run_jun = wind_speed_jun * wind_percent_time_jun * 0.44704 * 0.001
    wind_run_jul = wind_speed_jul * wind_percent_time_jul * 0.44704 * 0.001

    # INDICATOR FUNCTIONS

    # Create sI1 containing the indicator on whether target yard is affected by a V6-pathogen strain
    sI1_apr = np.array(data.loc[data['Month'] == 'April', 'Initial Strain'])
    sI1_may = np.array(data.loc[data['Month'] == 'May', 'Initial Strain'])
    sI1_jun = np.array(data.loc[data['Month'] == 'June', 'Initial Strain'])
    sI1_jul = np.array(data.loc[data['Month'] == 'July', 'Initial Strain'])

    # Create containing the constant array of 1's
    sI2 = np.ones((N,))

    # Create tI1 containing the indicator on whether target yard is only susceptible to V6-pathogen strain
    tI1 = np.zeros((N,))

    # Create tI2 containing the indicator on whether target yard is susceptible to both pathogen strains
    tI2 = np.zeros((N,))


    field_id = data['Field ID'].unique()

    for i in range(len(field_id)):

        sus_to_r6 = data.loc[(data['Month'] == 'May') & (data['Field ID'] == field_id[i]), 'Susceptibility to V6 Strains'].values[0]
        sus_to_nonr6 = data.loc[(data['Month'] == 'May') & (data['Field ID'] == field_id[i]), 'Susceptibility to non-V6 Strains'].values[0]
        
        # tI1
        
        if ((sus_to_r6 == 1) & (sus_to_nonr6 == 0)):  
            tI1[i] = 1
            
        else:
            tI1[i] = 0
        
        # tI2
            
        if ((sus_to_r6 == 1) & (sus_to_nonr6 == 1)):  
            tI2[i] = 1
        
        else:
            tI2[i] = 0
            
    # Function to convert bearing to angle in standard form
    def standard_form(angle):
        
        theta = np.pi / 2 - angle
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        theta = np.where(theta > 2 * np.pi, theta - 2 * np.pi, theta)
        
        return theta

    # Function to convert convert angle to angle between 0 and 2pi
    def coterminal(angle):
        
        theta = np.where(angle < 0, angle + 2 * np.pi, angle)
        theta = np.where(theta > 2 * np.pi, theta - 2 * np.pi, theta)
        
        return theta

    # Cardinal direction labels
    cardinal_directions = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])

    # Define 16 cardinal angles in radians
    cardinal_angles = np.linspace(0, 2 * np.pi, 17)
    cardinal_angles = cardinal_angles[:-1]

    # Convert bearing to angles in standard form
    cardinal_angles = standard_form(cardinal_angles)

    # Define geodesic shape of earth
    wgs84_geod = Geod(ellps='WGS84')

    # Create bearing_tensor containing the geodesic bearing between the great circles
    bearing_tensor = np.zeros((N, M))
    
    # Unique fields   
    field_id = data['Field ID'].unique()

    # Create distance containing geodesic distance between yard k and yard l
    distance = np.zeros((N, M))
    
    for i in range(N):
        for j in range(N):
            
            lat_i = data.loc[data['Field ID'] == field_id[i], 'Centroid Lat'].values[0]
            long_i = data.loc[data['Field ID'] == field_id[i], 'Centroid Long'].values[0]
            
            lat_j = data.loc[data['Field ID'] == field_id[j], 'Centroid Lat'].values[0]
            long_j = data.loc[data['Field ID'] == field_id[j], 'Centroid Long'].values[0]

            az_ij, az_jk, dist = wgs84_geod.inv(long_i, lat_i, long_j, lat_j)
            
            # Compute distance in km between yards k and l
            distance[i][j] = dist * 0.001
            
            # Convert azimuth to standard form
            az_ij = standard_form(az_ij * np.pi / 180)
            
            # Compute geodesic bearing from yard k to yard l
            if i == j:
                bearing_tensor[i][j] = np.nan
            else:
                bearing_tensor[i][j] = az_ij
    
    #Create adjacency matrix constraining spread between Non-R6 nodes and R6 nodes
    adjacency = np.zeros((N,M))
    
    
    # Create wind[i][j][k][l] containing mean of scalar projections of wind run
    # where i is the year, j is the month
    # k is the yard from which wind originates and l is the destination yard

    wind_apr = np.zeros((N, M))
    wind_may = np.zeros((N, M))
    wind_jun = np.zeros((N, M))
    wind_jul = np.zeros((N, M))
    

    for i in range(N):
        for j in range(M):

            # Define bounds within +/- 90 degrees of the bearing from yard i to yard j
            bounds = np.array([bearing_tensor[i][j] - np.pi / 2, bearing_tensor[i][j] + np.pi / 2])
            lower_bound = coterminal(min(bounds))
            upper_bound = coterminal(max(bounds))
            
            # Select cardinal directions within bounds
            cardinal_index = np.where(lower_bound < upper_bound, (lower_bound < cardinal_angles) & (cardinal_angles < upper_bound), (lower_bound < cardinal_angles) | (cardinal_angles < upper_bound))
            cardinal_index_num = np.where(cardinal_index)
            cardinal_index_num = cardinal_index_num[0]
            
            if len(cardinal_index_num) == 0:
                cardinal_index_num = [0] * 8
                
            # Compute bearing vector by first converting bearing to angles in standard position
            bearing_vector = np.array([np.cos(bearing_tensor[i][j]), np.sin(bearing_tensor[i][j])])
            
            # Select cardinal angles within bounds and convert to angles in standard position
            theta = cardinal_angles[cardinal_index]
            if len(theta) == 0:
                theta = [np.nan] * 8
            
            scalar_proj_list_apr = np.zeros((8, 1))
            scalar_proj_list_may = np.zeros((8, 1))
            scalar_proj_list_jun = np.zeros((8, 1))
            scalar_proj_list_jul = np.zeros((8, 1))
            
            for m in range(len(theta)):
                
                # APRIL
                
                # Compute wind vector
                wind_vector_apr = wind_run_apr[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_apr = wind_vector_apr.T
                
                wind_vector_apr = wind_run_apr[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_apr = wind_vector_apr.T
                
                wind_vector_apr = wind_run_apr[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_apr = wind_vector_apr.T
            
                # Compute scalar projection of wind_vector onto bearing_vector
                scalar_proj_apr = np.dot(wind_vector_apr, bearing_vector) / np.linalg.norm(bearing_vector)
                scalar_proj_list_apr[m] = scalar_proj_apr
                
                # MAY
                
                # Compute wind vector
                wind_vector_may = wind_run_may[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_may = wind_vector_may.T
                
                wind_vector_may = wind_run_may[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_may = wind_vector_may.T
                
                wind_vector_may = wind_run_may[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_may = wind_vector_may.T
            
                # Compute scalar projection of wind_vector onto bearing_vector
                scalar_proj_may = np.dot(wind_vector_may, bearing_vector) / np.linalg.norm(bearing_vector)
                scalar_proj_list_may[m] = scalar_proj_may
                
                # JUNE
                
                # Compute wind vector
                wind_vector_jun = wind_run_jun[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_jun = wind_vector_jun.T
                
                wind_vector_jun = wind_run_jun[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_jun = wind_vector_jun.T
                
                wind_vector_jun = wind_run_jun[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_jun = wind_vector_jun.T
            
                # Compute scalar projection of wind_vector onto bearing_vector
                scalar_proj_jun = np.dot(wind_vector_jun, bearing_vector) / np.linalg.norm(bearing_vector)
                scalar_proj_list_jun[m] = scalar_proj_jun
                
                # JULY
                
                # Compute wind vector
                wind_vector_jul = wind_run_jul[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_jul = wind_vector_jun.T
                
                wind_vector_jul = wind_run_jul[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_jul = wind_vector_jul.T
                
                wind_vector_jul = wind_run_jul[i][cardinal_index_num[m]] * np.array([np.cos(theta[m]), np.sin(theta[m])])
                wind_vector_jul = wind_vector_jul.T
            
                # Compute scalar projection of wind_vector onto bearing_vector
                scalar_proj_jul = np.dot(wind_vector_jul, bearing_vector) / np.linalg.norm(bearing_vector)
                scalar_proj_list_jul[m] = scalar_proj_jul
            
            
            # Compute mean of scalar projections    
            if i == j:
                wind_apr[i][j] = 0.0
                wind_may[i][j] = 0.0
                wind_jun[i][j] = 0.0
                wind_jul[i][j] = 0.0
            else:
                wind_apr[i][j] = np.sum(scalar_proj_list_apr) / len(scalar_proj_list_apr)
                wind_may[i][j] = np.sum(scalar_proj_list_may) / len(scalar_proj_list_may)
                wind_jun[i][j] = np.sum(scalar_proj_list_jun) / len(scalar_proj_list_jun)
                wind_jul[i][j] = np.sum(scalar_proj_list_jul) / len(scalar_proj_list_jul)
                
     
    # Save data 
                
    if yr == 2014:
        np.savez('../data/processed/data_2014', N=N, M=M, distance=distance, sI2=sI2, tI1=tI1, tI2=tI2, 
        y_apr=y_apr, n_apr=n_apr, a_apr=a_apr, wind_apr=wind_apr, sI1_apr=sI1_apr, s_apr=s_apr,
        y_may=y_may, n_may=n_may, a_may=a_may, wind_may=wind_may, sI1_may=sI1_may, s_may=s_may, 
        y_jun=y_jun, n_jun=n_jun, a_jun=a_jun, wind_jun=wind_jun, sI1_jun=sI1_jun, s_jun=s_jun,
        y_jul=y_jul, n_jul=n_jul, a_jul=a_jul, wind_jul=wind_jul, sI1_jul=sI1_jul, s_jul=s_jul)
        
    elif yr == 2015:
        np.savez('../data/processed/data_2015', N=N, M=M, distance=distance, sI2=sI2, tI1=tI1, tI2=tI2, 
        y_apr=y_apr, n_apr=n_apr, a_apr=a_apr, wind_apr=wind_apr, sI1_apr=sI1_apr, s_apr=s_apr,
        y_may=y_may, n_may=n_may, a_may=a_may, wind_may=wind_may, sI1_may=sI1_may, s_may=s_may, 
        y_jun=y_jun, n_jun=n_jun, a_jun=a_jun, wind_jun=wind_jun, sI1_jun=sI1_jun, s_jun=s_jun,
        y_jul=y_jul, n_jul=n_jul, a_jul=a_jul, wind_jul=wind_jul, sI1_jul=sI1_jul, s_jul=s_jul)

    elif yr == 2016:
        np.savez('../data/processed/data_2016', N=N, M=M, distance=distance, sI2=sI2, tI1=tI1, tI2=tI2, 
        y_apr=y_apr, n_apr=n_apr, a_apr=a_apr, wind_apr=wind_apr, sI1_apr=sI1_apr, s_apr=s_apr,
        y_may=y_may, n_may=n_may, a_may=a_may, wind_may=wind_may, sI1_may=sI1_may, s_may=s_may, 
        y_jun=y_jun, n_jun=n_jun, a_jun=a_jun, wind_jun=wind_jun, sI1_jun=sI1_jun, s_jun=s_jun,
        y_jul=y_jul, n_jul=n_jul, a_jul=a_jul, wind_jul=wind_jul, sI1_jul=sI1_jul, s_jul=s_jul)

    elif yr == 2017:
        np.savez('../data/processed/data_2017', N=N, M=M, distance=distance, sI2=sI2, tI1=tI1, tI2=tI2, 
        y_apr=y_apr, n_apr=n_apr, a_apr=a_apr, wind_apr=wind_apr, sI1_apr=sI1_apr, s_apr=s_apr,
        y_may=y_may, n_may=n_may, a_may=a_may, wind_may=wind_may, sI1_may=sI1_may, s_may=s_may, 
        y_jun=y_jun, n_jun=n_jun, a_jun=a_jun, wind_jun=wind_jun, sI1_jun=sI1_jun, s_jun=s_jun,
        y_jul=y_jul, n_jul=n_jul, a_jul=a_jul, wind_jul=wind_jul, sI1_jul=sI1_jul, s_jul=s_jul)
        
    # Print statement to confirm data was saved
    print(f'Processed data for year {yr} saved.')