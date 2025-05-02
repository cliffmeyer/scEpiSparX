import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def compute_regulatory_potential(cre, distances,  decay=10000, max_distance=100000):
    """
    Compute regulatory potential for each gene in each cell.

    Args:
        - cre: observed multiplier of activity (e.g., ATAC-seq signal) (cells x genes x cres)
        - distances: signed distances in kb from CRE to TSS (genes x cres)
        - decay: distance scaling factor
        - max_distance: maximum distance for regulatory influence

    Returns:
        - reg: regulatory potential for each gene in each cell (cells x genes)
    """
    eps = 1e-6  # Small constant to avoid division by zero
    num_cell_states, num_genes, _ = cre.shape
    reg = torch.zeros((num_cell_states, num_genes))  # Initialize output

    # Mask invalid distances and zero CRE activity
    mask = (distances != float('inf')) & (cre != 0)

    # Compute exponential decay term
    relative_dist = torch.clamp(torch.abs(distances), max=max_distance) / decay
    exp_term = torch.exp(-relative_dist)

    # Compute regulatory potential
    reg = (cre * exp_term).sum(dim=-1)  

    return reg


def compute_gene_rp_correlations(gene_ex, rp):
    """
    Compute the correlation between gene expression and regulatory potential for each gene.
    
    Args:
        - gene_ex: Tensor of gene expression values (cells x genes)
        - rp: Tensor of regulatory potential values (cells x genes)
    
    Returns:
        - corrs: Tensor of correlation coefficients for each gene (genes,)
        - mean_corr: Mean correlation across all genes
        - var_corr: Variance of correlations across all genes
    """
    num_genes = gene_ex.shape[1]
    num_cells = gene_ex.shape[0]
    corrs = torch.zeros(num_genes)

    for i in range(num_genes):
        x = gene_ex[:,i]  # Expression for gene i across cells
        y = rp[:,i]  # Regulatory potential for gene i across cells

        # Compute Pearson correlation
        if torch.std(x) > 0 and torch.std(y) > 0:
            corrs[i] = torch.corrcoef(torch.stack([x, y]))[0, 1]
        else:
            corrs[i] = float('nan')  # Assign NaN if std is zero to avoid errors

    # Compute mean and variance, ignoring NaNs
    valid_corrs = corrs[~torch.isnan(corrs)]
    mean_corr = valid_corrs.mean().item()
    var_corr = valid_corrs.var().item()

    valid_corrs = corrs[~torch.isnan(corrs)].cpu().numpy()  # Remove NaNs and convert to NumPy
    plt.figure(figsize=(8, 5))
    plt.hist(valid_corrs, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.title("Histogram of Gene Expression vs Regulatory Potential Correlations")
    plt.grid(True)
    plt.show()

    # Generate scatter plots separately for each cell type
    for i in range(num_cells):
        plt.figure(figsize=(6, 6))
        plt.scatter(gene_ex[i].detach().cpu().numpy(),
                    rp[i].detach().cpu().numpy(),
                    alpha=0.3, color="blue", s=10)
        
        plt.xlabel("Gene Expression")
        plt.ylabel("Regulatory Potential")
        plt.title(f"RP vs. Gene Expression (Cell Type {i})")
        plt.show()

    return corrs, mean_corr, var_corr


