from joblib._utils import limit
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import contextily as cx

#Read in Data and predictions
data = pd.read_csv("data/processed/cost_data.csv")
preds = pd.read_csv("data/processed/results/mle_preds.csv")
edge_weights = pd.read_csv("data/processed/results/mle_edges_long.csv")

#Subset unique coordinates
coords = data[['Field ID', 'Centroid Lat', 'Centroid Long']].drop_duplicates(subset = ['Field ID'], keep = 'first')
coords.columns = ['field_id', 'lat', 'long']

#Join to predictions
preds_join = pd.merge(preds, coords, left_on='field_id', right_on='field_id', how='inner')

sp_preds = gpd.GeoDataFrame(preds_join, geometry=gpd.points_from_xy(preds_join.long, preds_join.lat), crs = "EPSG:4326")

ax = sp_preds[(sp_preds['month'] == 'jul') & (sp_preds['year'] == 2014) & (sp_preds['scenario'] == 'towards_zero')].plot('deviance_resid', alpha = 0.75, legend=True)
cx.add_basemap(ax, crs = sp_preds.crs, source="OpenStreetMap.Mapnik")
