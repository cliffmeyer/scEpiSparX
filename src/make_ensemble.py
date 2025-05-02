import os
import torch
import numpy as np
from pathlib import Path

from model_embed import MIRA_Poisson_Regressor
from scipy.optimize import linear_sum_assignment

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def load_model(model_path):

    print( model_path )

    if os.path.exists(model_path):
        _dict = torch.load(model_path)
        print(_dict["params"])
        print(_dict["state_dict"])

        model = MIRA_Poisson_Regressor(**_dict["params"])
        model.load_state_dict(_dict["state_dict"])
        model.eval()  # Set to evaluation mode
        print(f"Loaded existing model: {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model path {model_path} not found.")


def save_model(model, path='model.pth'):
    """
    Save the model with both parameters and state_dict.

    Parameters:
        model: Trained model instance.
        path (str): Path to save the model file.
    """
    model_params = { 
        "num_genes": model.num_genes,
        "max_cres_per_gene": model.max_cres_per_gene,
        "num_cell_states": model.num_cell_states,
        "num_pro_types": model.num_pro_types,
        "num_enh_types": model.num_enh_types,
        "num_classes": model.num_classes,
        "embedding_dim": model.embedding_dim,
        "use_embeddings": model.use_embeddings
    }   
    torch.save({"state_dict": model.state_dict(), "params": model_params}, path)
    print(f"Model saved to {path}")


def update_embeddings(model, pro_embed, enh_embed, save_path="updated_model.pth"):
    """
    Update the model's promoter and enhancer embedding weights and save the model.

    Parameters:
        model: The model with embedding_to_pro_type and embedding_to_enh_type layers.
        pro_embed (torch.Tensor or np.ndarray): New promoter embeddings.
        enh_embed (torch.Tensor or np.ndarray): New enhancer embeddings.
        save_path (str): File path to save the updated model.
    """
    if isinstance(pro_embed, np.ndarray):
        pro_embed = torch.tensor(pro_embed, dtype=model.embedding_to_pro_type.weight.dtype)
    if isinstance(enh_embed, np.ndarray):
        enh_embed = torch.tensor(enh_embed, dtype=model.embedding_to_enh_type.weight.dtype)

    model.embedding_to_pro_type.weight.data = pro_embed
    model.embedding_to_enh_type.weight.data = enh_embed

    save_model(model, path=save_path)


def extract_embeddings(model):
    pro_embed = model.embedding_to_pro_type.weight.detach().numpy()
    enh_embed = model.embedding_to_enh_type.weight.detach().numpy()
    return pro_embed, enh_embed


def extract_acts(model):
    pro_act = model.pro_act.data.clone().cpu().numpy()
    enh_act = model.enh_act.data.clone().cpu().numpy()
    return pro_act, enh_act


def assign_embeddings_to_cells(act):
    """
    args:
        - act: num_embedding_vectors x num_cell_types matrix
    returns:
        - list of tuples assigning embeddings to cell types 
    """
    # Take abs and negate for maximization
    cost_matrix = -np.abs(act)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    max_total = cost_matrix[row_indices, col_indices].sum()
    assignments = list(zip(row_indices, col_indices))
    print("Assignments (row to column):", assignments)
    print("Maximum total value:", max_total)
    return assignments


def run_leiden_clustering(X, n_neighbors=15, resolution=1.0):
    """
    Run Leiden clustering on a matrix of row vectors using cosine similarity,
    and return cluster centroids.

    Parameters:
        X (np.ndarray): Rows = samples, columns = features.
        n_neighbors (int): Number of neighbors for the graph.
        resolution (float): Resolution parameter for Leiden clustering.

    Returns:
        adata (AnnData): Annotated data object with Leiden labels in `adata.obs['leiden']`
        centroids (np.ndarray): Rows = cluster centroids.
    """
    # Normalize row vectors
    X_normalized = normalize(X, axis=1)

    # Create AnnData
    adata = ad.AnnData(X_normalized)

    # Build graph and run Leiden
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric='cosine', use_rep='X')
    sc.tl.leiden(adata, resolution=resolution)

    # Compute centroids
    labels = adata.obs['leiden'].astype(int).values
    unique_labels = np.unique(labels)
    centroids = np.vstack([
        X_normalized[labels == label].mean(axis=0)
        for label in unique_labels
    ])

    return adata, centroids

def run_leiden_with_target_clusters(X, target_clusters, n_neighbors=10, 
                                     initial_resolution=1.0, tol=0.01, max_iter=20):
    """
    Run Leiden clustering and automatically adjust resolution until 
    the number of clusters matches target_clusters.

    Parameters:
        X (np.ndarray): Data matrix (rows = samples).
        target_clusters (int): Desired number of clusters (e.g. n_enh_embeds).
        n_neighbors (int): Neighbors for graph construction.
        initial_resolution (float): Starting resolution.
        tol (float): Tolerance for resolution step size.
        max_iter (int): Maximum number of search iterations.

    Returns:
        adata: AnnData with Leiden clustering.
        centroids: np.ndarray of cluster centroids (rows = clusters).
    """
    low, high = 0.01, 10.0
    resolution = initial_resolution
    for i in range(max_iter):
        print( '__ resolution:', resolution )
        adata, centroids = run_leiden_clustering(X, n_neighbors=n_neighbors, resolution=resolution)
        n_clusters = centroids.shape[0]

        if n_clusters == target_clusters:
            break
        elif n_clusters < target_clusters:
            low = resolution
        else:
            high = resolution

        resolution = (low + high) / 2
        if abs(high - low) < tol:
            break

    print(f"Final resolution: {resolution:.4f}, clusters: {n_clusters}")
    return adata, centroids


def plot_umap(adata, color='leiden'):
    """
    Compute and plot UMAP colored by Leiden cluster.

    Parameters:
        adata (AnnData): AnnData object with neighbors and clustering done.
        color (str): Column in `adata.obs` to color by (default is 'leiden').
    """
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=color, frameon=False, show=True)


def main():
    model_paths = sorted(Path("./").glob("*.pth"))

    enh_embeds = []
    pro_embeds = []
    for path in model_paths:
        model = load_model(path)
        pro_act, enh_act = extract_acts(model)
        _pro_embed, _enh_embed = extract_embeddings(model)
        print( 'pro shape', pro_act.shape )
        print( 'enh shape', enh_act.shape )
        pro_assignments = assign_embeddings_to_cells(pro_act)
        enh_assignments = assign_embeddings_to_cells(enh_act)
        print("Type assignments for cell types:", pro_assignments)
        print("Type assignments for cell types:", enh_assignments)
        _pro_idx = [ elem[0] for elem in pro_assignments ]
        _enh_idx = [ elem[0] for elem in enh_assignments ]
        pro_embeds.append(_pro_embed[_pro_idx])
        enh_embeds.append(_enh_embed[_enh_idx])

    n_pro_embeds = pro_act.shape[0]
    n_enh_embeds = enh_act.shape[0]

    pro_embeds = np.vstack(pro_embeds)
    adata, pro_embed_centroids = run_leiden_with_target_clusters(pro_embeds, target_clusters=n_pro_embeds)
    plot_umap(adata, color='leiden')

    enh_embeds = np.vstack(enh_embeds)
    adata, enh_embed_centroids = run_leiden_with_target_clusters(enh_embeds, target_clusters=n_enh_embeds)
    plot_umap(adata, color='leiden')

    update_embeddings(model, pro_embed_centroids, enh_embed_centroids, save_path="updated_model.pth")


if __name__ == "__main__":
    main()


