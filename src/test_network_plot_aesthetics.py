import os
import cmdstanpy
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# --- AESTHETIC AND DATA CONFIGURATION ---
# Change these values to adjust the plot's data source and appearance.
model = "binomial"
month = "jul"
year = 2014

# Edge filtering configuration.
EDGE_FILTER_MODE = "percentile"   # "percentile" or "top_k"
EDGE_PERCENTILE = 98             # used if mode == "percentile"
EDGE_TOP_K = 3                    # used if mode == "top_k"

# Figure and plotting aesthetics.
FIG_SIZE = (14, 12)
NODE_SIZE = 100
NODE_ALPHA = 0.75
NODE_CMAP = plt.cm.viridis
EDGE_CMAP = plt.cm.YlOrRd
ARROW_SIZE = 10
CONNECTION_STYLE = "arc3,rad=0.15"
BASEMAP_SOURCE = cx.providers.CartoDB.Positron

# Load and preprocess data stacked across years and by year.
def load_and_preprocess_hop_data(years, data_path_template='data/processed/data_{year}.npz'):
    data_by_year = {}
    vector_keys = [
        'field_id', 'year_vec', 'tI1', 'y_apr', 'y_may', 'y_jun', 'y_jul',
        'n_apr', 'n_may', 'n_jun', 'n_jul',
        'a_apr', 'a_may', 'a_jun', 'a_jul',
        'sI1_apr', 'sI1_may', 'sI1_jun', 'sI1_jul',
        's_apr', 's_may', 's_jun', 's_jul'
    ]
    matrix_keys = ['distance', 'wind_apr', 'wind_may', 'wind_jun', 'wind_jul']

    for y in years:
        path = data_path_template.format(year=y)
        try:
            raw_data = np.load(path)
            N_yr = int(raw_data['N'])
            processed = {'N': N_yr}
            for key in vector_keys:
                if key in raw_data:
                    processed[key] = raw_data[key].reshape(N_yr, 1)
            for key in matrix_keys:
                if key in raw_data:
                    processed[key] = raw_data[key]
            data_by_year[y] = processed
        except FileNotFoundError:
            continue
    
    stacked_data = {}
    for key in vector_keys:
        available_arrays = [data_by_year[y][key] for y in data_by_year if key in data_by_year[y]]
        if available_arrays:
            stacked_data[key] = np.vstack(available_arrays)

    return data_by_year, stacked_data

# Prepare input data for Stan.
def prepare_stan_inputs(analysis_month, data_by_year, stacked_data, years, prior_config):
    month_map = {'may': 'apr', 'jun': 'may', 'jul': 'jun'}
    lag_month = month_map[analysis_month]
    N_max = int(max(data_by_year[y]['N'] for y in years))
    T = len(years)
    
    year_starts, year_ends, year_sizes = [], [], []
    current_idx = 1 
    
    for y in years:
        N_yr = int(data_by_year[y]['N'])
        year_sizes.append(N_yr)
        year_starts.append(current_idx)
        year_ends.append(current_idx + N_yr - 1)
        current_idx += N_yr

    dist_mats = np.zeros((T, N_max, N_max)) 
    wind_mats = np.zeros((T, N_max, N_max))

    for i, y in enumerate(years):
        N_yr = int(data_by_year[y]['N'])
        dist_mats[i, :N_yr, :N_yr] = data_by_year[y]['distance']
        wind_mats[i, :N_yr, :N_yr] = data_by_year[y][f'wind_{lag_month}']

    factorized_ids, unique_ids = pd.factorize(stacked_data['field_id'].flatten())

    stan_data = {
        "T": T, "N_total": sum(year_sizes), "N_max": N_max, 
        "N_yards": len(unique_ids),
        "yard_ids": (factorized_ids + 1).tolist(), 
        "year_starts": year_starts, "year_ends": year_ends, "year_sizes": year_sizes,
        "y": stacked_data[f'y_{analysis_month}'].flatten().astype(int).tolist(), 
        "n": stacked_data[f'n_{analysis_month}'].flatten().astype(int).tolist(),
        "y_lag": stacked_data[f'y_{lag_month}'].flatten(),
        "n_lag": stacked_data[f'n_{lag_month}'].flatten(),
        "s_lag": stacked_data[f's_{lag_month}'].flatten(),
        "sI1_lag": stacked_data[f'sI1_{lag_month}'].flatten(),
        "a_lag": stacked_data[f'a_{lag_month}'].flatten(),
        "cultivar": stacked_data['tI1'].flatten(),
        "dist_mats": dist_mats, "wind_mats": wind_mats,
        "J": len(np.unique(stacked_data['field_id'].flatten())),
        **prior_config 
    }
    return stan_data

# Set years and prior configuration.
years = [2014, 2015, 2016, 2017]
prior_config = {
    "beta_mu": 0,       "beta_sigma": 2.5, 
    "delta_mu": 0,      "delta_sigma": 2.5, 
    "gamma_mu": 0,      "gamma_sigma": 2.5, 
    "alpha_mu": 0,      "alpha_sigma": 2.5, 
    "eta1_mu": 0,       "eta1_sigma": 2.5, 
    "eta2_mu": 0,       "eta2_sigma": 2.5,
    "phi_mu": 0,        "phi_sigma": 2.5
}

# Load the raw, preprocessed, and stacked data.
data_by_year, stacked = load_and_preprocess_hop_data(years)
hop_pm = pd.read_csv("data/processed/cost_data.csv")
stacked_df = pd.read_csv("data/processed/stacked_data.csv")

# Extract and clean unique field coordinates lookup.
coords = hop_pm[['Field ID', 'Centroid Lat', 'Centroid Long']].drop_duplicates(subset=['Field ID'], keep='first')
coords.columns = ['field_id', 'lat', 'long']
coords['field_id'] = coords['field_id'].astype(str)

# Prepare Stan inputs and retrieve unique field IDs.
stan_data = prepare_stan_inputs(month, data_by_year, stacked, years, prior_config)
_, unique_ids = pd.factorize(stacked['field_id'].flatten())

# Load MCMC fit outputs from CSV files.
fit_dir = f"results/stan_fits/{model}/{month}"
fit = cmdstanpy.from_csv(fit_dir)

# Extract full posterior matrix draws.
all_edges = fit.stan_variable("edge_weights")
logit_p_draws = fit.stan_variable("logit_p")

mean_matrix_padded_all = np.mean(all_edges, axis=0)
mean_logit_p_all = np.mean(logit_p_draws, axis=0)

# Get index bounds for the current year from stacked_data.csv.
year_indices = stacked_df[stacked_df['year_vec'] == year].index
start_idx = year_indices.min()
end_idx = year_indices.max() + 1
N_yr = end_idx - start_idx

# Extract year-specific adjacency matrix slice and remove self-loops.
adj_matrix = mean_matrix_padded_all[start_idx:end_idx, start_idx:end_idx].copy()
np.fill_diagonal(adj_matrix, 0)

# Construct directed graph and relabel nodes with factorized IDs.
active_factorized_ids = stan_data['yard_ids'][start_idx:end_idx]
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
mapping = {i: fid for i, fid in enumerate(active_factorized_ids)}
G = nx.relabel_nodes(G, mapping)
out_strength = dict(G.out_degree(weight='weight'))

# Calculate risk probabilities.
mean_logit_p_active = mean_logit_p_all[start_idx:end_idx]
estimated_probabilities = 1.0 / (1.0 + np.exp(-mean_logit_p_active))

# Merge nodes with coordinates.
nodes_df = pd.DataFrame({
    'factorized_id': active_factorized_ids,
    'field_id': [str(unique_ids[fid - 1]) for fid in active_factorized_ids],
    'mcmc_prob': estimated_probabilities,
    'out_strength': [out_strength[fid] for fid in active_factorized_ids]
})
nodes_df = pd.merge(nodes_df, coords, on='field_id', how='inner')

# Convert node dataframe to GeoDataFrame.
sp_nodes = gpd.GeoDataFrame(
    nodes_df,
    geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat),
    crs="EPSG:4326"
)

# Position dictionary for plotting nodes.
pos = {row.factorized_id: (row.long, row.lat) for row in nodes_df.itertuples()}
G_sub = G.subgraph(pos.keys())

# Filter edges to reduce visual clutter.
all_weights_this_year = [w for _, _, w in G_sub.edges(data='weight')]
G_plot = nx.DiGraph()
G_plot.add_nodes_from(G_sub.nodes(data=True))

if EDGE_FILTER_MODE == "percentile" and all_weights_this_year:
    cutoff = np.percentile(all_weights_this_year, EDGE_PERCENTILE)
    G_plot.add_edges_from(
        (u, v, d) for u, v, d in G_sub.edges(data=True) if d['weight'] >= cutoff
    )
elif EDGE_FILTER_MODE == "top_k":
    for node in G_sub.nodes():
        out_edges = sorted(G_sub.out_edges(node, data=True), key=lambda e: e[2]['weight'], reverse=True)
        G_plot.add_edges_from(out_edges[:EDGE_TOP_K])
else:
    G_plot = G_sub

# Create figure.
fig, ax = plt.subplots(figsize=FIG_SIZE)

# Layer 1: Plot nodes.
sp_nodes.plot(
    ax=ax,
    column='out_strength',
    alpha=NODE_ALPHA,
    legend=True,
    cmap=NODE_CMAP,
    markersize=NODE_SIZE,
    legend_kwds={'label': "Outgoing Degree Centrality"}
)
ax.collections[-1].set_zorder(2)

# Layer 2: Draw transmission flux edges.
edges_in_map = list(G_plot.edges())
weights = [G_plot[u][v]['weight'] for u, v in edges_in_map]

if len(weights) > 0:
    max_w = max(weights) if max(weights) > 0 else 1
    display_widths = [1.5 + (w / max_w) * 6.5 for w in weights]

    edge_artists = nx.draw_networkx_edges(
        G_plot, pos, ax=ax,
        edgelist=edges_in_map,
        width=2.5,
        edge_color=weights,
        edge_cmap=EDGE_CMAP,
        edge_vmin=min(weights),
        edge_vmax=max(weights),
        arrowstyle="->",
        arrowsize=ARROW_SIZE,
        connectionstyle=CONNECTION_STYLE,
        node_size=0
    )
    if isinstance(edge_artists, list):
        for artist in edge_artists:
            artist.set_zorder(3)
    else:
        edge_artists.set_zorder(3)

    # Draw colorbar for edge weights.
    sm = plt.cm.ScalarMappable(
        cmap=EDGE_CMAP,
        norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
    )
    sm.set_array([])
    fig.colorbar(
        sm, ax=ax,
        orientation='horizontal',
        location='bottom',
        shrink=0.6,
        pad=0.03,
        label="Edge Weight (Posterior Mean)"
    )

# Layer 3: Add basemap.
cx.add_basemap(ax, crs=sp_nodes.crs, source=BASEMAP_SOURCE)
ax.images[-1].set_zorder(1)

# Set axes limits, titles, layout, and save.
ax.set_aspect('equal')
plt.title(f"Hop Powdery Mildew Transmission Network — Outgoing Node Strength — {month.capitalize()} {year} ({model})", fontsize=14)
plt.axis('off')
plt.tight_layout()

# Save figure to file.
os.makedirs("output/figures", exist_ok=True)
plot_path = "output/figures/test_network_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {plot_path}")
plt.show()
