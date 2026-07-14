import cmdstanpy
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ==========================================
# 1. Read in Raw Data and Define Pipeline
# ==========================================
data_by_year = np.load("data/processed/data_by_year.npz", allow_pickle=True)['arr_0'].item()
stacked = np.load("data/processed/stacked_data.npz", allow_pickle=True)['arr_0'].item()
hop_pm = pd.read_csv("data/processed/cost_data.csv")

# Clean unique coordinates lookup from your raw dataframe
coords = hop_pm[['Field ID', 'Centroid Lat', 'Centroid Long']].drop_duplicates(subset=['Field ID'], keep='first')
coords.columns = ['field_id', 'lat', 'long']
# Normalize the join key to string once, up front, so every downstream join
# (network nodes AND geodataframe merge) agrees on what counts as a match.
coords['field_id'] = coords['field_id'].astype(str)


def prepare_stan_inputs(analysis_month, data_by_year, stacked_data, years):
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
        "J": len(np.unique(stacked_data['field_id'].flatten()))
    }
    return stan_data, unique_ids


# Load outputs
month = "jul"
years = list(range(2014, 2018))
fit = cmdstanpy.from_csv(f"results/stan_fits/{month}/weakly_informative")

stan_data, unique_ids = prepare_stan_inputs(month, data_by_year, stacked, years=years)

# Extract full posterior matrices from your fit object
all_edges = fit.stan_variable("edge_weights")
logit_p_draws = fit.stan_variable("logit_p")

mean_matrix_padded_all = np.mean(all_edges, axis=0)
width_matrix_padded_all = np.quantile(all_edges, 0.95, axis=0) - np.quantile(all_edges, 0.05, axis=0)
mean_logit_p_all = np.mean(logit_p_draws, axis=0)

# ==========================================
# 1b. Edge Filtering Config
# ==========================================
# The transmission network is effectively fully connected (every field has some
# nonzero estimated flux to every other field), so drawing every edge produces
# an unreadable hairball. Choose ONE of the two filtering strategies below.
EDGE_FILTER_MODE = "percentile"   # "percentile" or "top_k"
EDGE_PERCENTILE = 95             # only keep edges >= this percentile of weight (used if mode == "percentile")
EDGE_TOP_K = 3                    # keep each node's k strongest outgoing edges (used if mode == "top_k")

# ==========================================
# 2. Loop Through Years and Generate Plots
# ==========================================

for t_idx, year in enumerate(years):
    print(f"Processing spatial network visualization for year: {year}...")

    N_yr = stan_data['year_sizes'][t_idx]
    start_idx = stan_data['year_starts'][t_idx] - 1
    end_idx = stan_data['year_ends'][t_idx]

    # --------------------------------------
    # A. Process Adjacency Grid Slice
    # --------------------------------------
    adj_matrix = mean_matrix_padded_all[t_idx, 0:N_yr, 0:N_yr].copy()
    # Guard against self-loops: if the diagonal isn't exactly zero, from_numpy_array
    # will create edges from a node back to itself, which render as stray loops.
    np.fill_diagonal(adj_matrix, 0)

    active_factorized_ids = stan_data['yard_ids'][start_idx:end_idx]

    # Construct Directed Graph
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Relabel sequence nodes to match the factorized Stan IDs
    mapping = {i: fid for i, fid in enumerate(active_factorized_ids)}
    G = nx.relabel_nodes(G, mapping)

    # Outgoing node strength: sum of outgoing edge weights for each field (i.e.
    # weighted out-degree, NOT the normalized 0-1 centrality metric). Computed on
    # the FULL per-year graph, before subsetting to coordinate-matched fields or
    # applying the display edge filter, so it reflects true network structure.
    out_strength = dict(G.out_degree(weight='weight'))

    # --------------------------------------
    # B. Calculate Risk Probabilities via Inverse-Logit
    # --------------------------------------
    mean_logit_p_active = mean_logit_p_all[start_idx:end_idx]
    estimated_probabilities = 1.0 / (1.0 + np.exp(-mean_logit_p_active))

    nodes_df = pd.DataFrame({
        'factorized_id': active_factorized_ids,
        'field_id': [str(unique_ids[fid - 1]) for fid in active_factorized_ids],
        'mcmc_prob': estimated_probabilities,
        'out_strength': [out_strength[fid] for fid in active_factorized_ids]
    })
    # Single, consistent join key (string) used for BOTH node coordinates and
    # network positions below, so the map dots and the graph edges are always
    # built from the same set of matched fields.
    nodes_df = pd.merge(nodes_df, coords, on='field_id', how='inner')

    sp_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat),
        crs="EPSG:4326"
    )

    # Build node positions directly from the merged (and therefore
    # coordinate-confirmed) dataframe, keyed by factorized_id to match G's nodes.
    pos = {
        row.factorized_id: (row.long, row.lat)
        for row in nodes_df.itertuples()
    }

    # Restrict the graph to nodes we actually have a position for, so
    # draw_networkx_edges never hits a KeyError on a field with no coordinates.
    G_sub = G.subgraph(pos.keys())

    # --- DIAGNOSTICS: remove once edges are confirmed showing up ---
    print(f"  [{year}] adj_matrix: nonzero entries = {np.count_nonzero(adj_matrix)}, "
          f"max = {adj_matrix.max():.4g}")
    print(f"  [{year}] G (pre-subgraph): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  [{year}] matched coords: {len(pos)} of {len(active_factorized_ids)} active fields")
    print(f"  [{year}] G_sub (post-subgraph): {G_sub.number_of_nodes()} nodes, "
          f"{G_sub.number_of_edges()} edges")
    # -----------------------------------------------------------------

    # Reduce visual clutter: keep only the strongest edges per EDGE_FILTER_MODE.
    all_weights_this_year = [w for _, _, w in G_sub.edges(data='weight')]
    if EDGE_FILTER_MODE == "percentile" and all_weights_this_year:
        cutoff = np.percentile(all_weights_this_year, EDGE_PERCENTILE)
        G_plot = nx.DiGraph()
        G_plot.add_nodes_from(G_sub.nodes(data=True))
        G_plot.add_edges_from(
            (u, v, d) for u, v, d in G_sub.edges(data=True) if d['weight'] >= cutoff
        )
    elif EDGE_FILTER_MODE == "top_k":
        G_plot = nx.DiGraph()
        G_plot.add_nodes_from(G_sub.nodes(data=True))
        for node in G_sub.nodes():
            out_edges = sorted(G_sub.out_edges(node, data=True), key=lambda e: e[2]['weight'], reverse=True)
            G_plot.add_edges_from(out_edges[:EDGE_TOP_K])
    else:
        G_plot = G_sub

    print(f"  [{year}] G_plot (after {EDGE_FILTER_MODE} filter): {G_plot.number_of_edges()} edges "
          f"(kept from {G_sub.number_of_edges()})")



    # --------------------------------------
    # C. Render Current Year Map Canvas (Explicit Layering Order)
    # --------------------------------------
    fig, ax = plt.subplots(figsize=(14, 12))

    # LAYER 1 (drawn first, but rendered on the bottom via zorder):
    # Plot the field dots FIRST so the axes autoscale to the correct
    # geographic extent. contextily.add_basemap reads the *current* axes
    # extent to pick which tiles to fetch -- if it's called on an empty
    # axes, it fetches tiles for the default (0,1)x(0,1) view instead of
    # your actual field locations.
    sp_nodes.plot(
        ax=ax,
        column='out_strength',
        alpha=0.9,
        legend=True,
        cmap='viridis',
        markersize=180,
        # No fixed vmin/vmax: out_strength isn't bounded like a probability,
        # so let it auto-scale to this year's actual range.
        legend_kwds={'label': "Outgoing Degree Centrality"}
    )
    # GeoDataFrame.plot() doesn't take a zorder kwarg reliably across versions and
    # returns the Axes, not the artist -- so grab the collection it just added.
    ax.collections[-1].set_zorder(2)

    # LAYER 2: NetworkX Transmission Flux Paths
    edges_in_map = list(G_plot.edges())
    weights = [G_plot[u][v]['weight'] for u, v in edges_in_map]

    if len(weights) > 0:
        max_w = max(weights) if max(weights) > 0 else 1
        # Minimum visibility width 1.5 px, scaling up to 8.0 px for the strongest links
        display_widths = [1.5 + (w / max_w) * 6.5 for w in weights]

        edge_artists = nx.draw_networkx_edges(
            G_plot, pos, ax=ax,
            edgelist=edges_in_map,
            width=2,
            edge_color=weights,
            edge_cmap=plt.cm.berlin,  # Dark blue (low flux) -> white -> deep red (high flux)
            edge_vmin=min(weights),
            edge_vmax=max(weights),
            arrowstyle="->",
            arrowsize=14,
            connectionstyle="arc3,rad=0.15",
            node_size=0
        )
        # draw_networkx_edges has no zorder kwarg. For a DiGraph it returns a list
        # of FancyArrowPatch objects; for a Graph it returns a single LineCollection.
        # Handle both so this still works if the graph type ever changes.
        if isinstance(edge_artists, list):
            for artist in edge_artists:
                artist.set_zorder(3)
        else:
            edge_artists.set_zorder(3)

        # Colorbar for edge weights, since edge_cmap doesn't get one automatically.
        # Placed at the bottom (horizontal) so it doesn't stack up against the
        # node-strength colorbar on the right.
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.berlin,
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

    # LAYER 3: Basemap drawn last but placed at the bottom of the stack via zorder.
    # Now that sp_nodes.plot() has set the axes extent, the fetched tiles will
    # actually cover your field locations.
    cx.add_basemap(ax, crs=sp_nodes.crs, source=cx.providers.CartoDB.Positron)
    ax.images[-1].set_zorder(1)

    # Enforce geographic configuration parameters
    ax.set_aspect('equal')
    plt.title(f"Hop Powdery Mildew Transmission Network — Outgoing Node Strength — July {year}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()