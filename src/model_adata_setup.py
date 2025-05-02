import anndata as ad
import csv
import json
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import embedding_utils

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import pearsonr
from typing import Tuple, List

PROXIMAL_RANGE = 1000  # Define promoter region distance threshold
PROXIMAL_INDEX, UPSTREAM_INDEX, DOWNSTREAM_INDEX = 0, 1, 2  # CRE class labels

ALIAS_FILE = "/Users/len/Projects/cistromesparx_dev/data/gene_regulation/hgnc_complete_set.json"
TSS_FILE = "/Users/len/Projects/lisa2/lisa2/lisa/genomes/hg38.refseq"
EMBED_FILE = "/Users/len/Projects/cistromesparx_dev/results/sparx/ATAC_HG38/ATAC_HG38_regions.h5"

def read_alias_file(filename):
    """
    Read in previous symbols for gene and return dict mapping previous symbols to current.
    """
    with open(filename) as fp:
        gene_info = json.load(fp)
    gene_info = gene_info['response']['docs']
    d = {}
    for gene in gene_info:
        if 'prev_symbol' in gene:
            t = {k:gene['symbol'] for k in gene['prev_symbol'] }
            d.update(t)
    return d


def read_gene_table(filename):
    """
    read in file of refseq genes and return dict mapping gene symbol to TSS coordinate.
    args:
        - filename (str): refseq tab delimited file with columns defined as in HEADER
    returns:
        (dict) keys: gene symbols, values: TSS (chrom,coord,strand)
    """
    HEADER = ['refseq_id','chrom','strand','start','end','exon_s','exon_e','sym']

    header_idx = { k:i for i,k in enumerate(HEADER) }
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter='\t')
        data = list(reader)
        gene_lookup = {}
        for row in data:
            sym = row[ header_idx['sym'] ]
            strand = row[ header_idx['strand'] ]
            if strand == '+':
                pos_tss = ( row[ header_idx['chrom'] ], int(row[ header_idx['start'] ]), strand )
            else:
                pos_tss = ( row[ header_idx['chrom'] ], int(row[ header_idx['end'] ]), strand )
            gene_lookup[sym] = pos_tss

    return gene_lookup


def annotate_gene_tss(adata, gene_alias_file=ALIAS_FILE, gene_ref_file=TSS_FILE):
    """
    Annotate genes in adata with TSS coordinates.
    
    Args:
        - adata (AnnData): Single-cell or genomic data stored in an AnnData object.
        - gene_alias_file (str): Path to the gene alias file (HGNC JSON format).
        - gene_ref_file (str): Path to the RefSeq gene table file.
    
    Modifies:
        - Adds 'tss' column in `adata.var`, mapping gene symbols to their (chrom, TSS) coordinates.
    """
    _CHROM,_COORD,_STRAND=0,1,2
    gene_alias = read_alias_file(gene_alias_file)
    gene_lookup = read_gene_table(gene_ref_file)

    # Map gene names to current symbols
    genes = adata.var_names.to_list()
    updated_genes = [gene_alias.get(g, g) for g in genes]

    # Assign TSS coordinates based on updated gene names
    tss_annotations = [gene_lookup.get(g, (None, None, None)) for g in updated_genes]

    # Store in adata.var
    adata.var['chrom'] = [t[_CHROM] for t in tss_annotations]
    adata.var['tss'] = [t[_COORD] for t in tss_annotations]
    adata.var['strand']    = [t[_STRAND] for t in tss_annotations]

    return adata


def annotate_gene_symbols(adata):
    """
    Extracts gene symbols from the 'gene_ids' column in an AnnData object
    and adds them as a new column 'gene_symbols' in adata.var.

    Args:
        adata (AnnData): AnnData object containing gene information.

    Modifies:
        - Adds a new column `gene_symbols` to `adata.var`.
    """
    adata.var["gene_symbols"] = adata.var.index.astype(str)
    return adata


def plot_reg_potential_vs_expression_per_cell(gex_adata, max_cells=10):
    """
    Plots scatter plots of regulatory potential vs. gene expression for all genes in each cell.
    
    Parameters:
    - gex_adata: AnnData object containing gene expression and regulatory potential (cells x genes).
    - max_cells: Maximum number of cells to plot (for visualization clarity).
    """
    num_cells = min(gex_adata.n_obs, max_cells)  # Limit plots for readability
    gene_names = gex_adata.var["gene_symbols"].values
    
    for cell_idx in range(num_cells):
        gene_expression = gex_adata.X[cell_idx, :].ravel()
        regulatory_potential = gex_adata.varm['regulatory_potential'][:, cell_idx].ravel()
        
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=regulatory_potential, y=gene_expression, alpha=0.6)
        plt.xlabel("Regulatory Potential")
        plt.ylabel("Gene Expression")
        plt.title(f"Cell {cell_idx}: Reg. Pot. vs Expression")
        plt.grid(True)
        plt.show()


def parse_cre_genomic_coords(adata):
    """
    Annotate CREs with genomic coordinates, filtering out invalid regions.

    Args:
        - adata (AnnData): Single-cell ATAC-seq data stored in an AnnData object.

    Modifies:
        - Adds 'chrom', 'start', and 'end' columns to adata.var, keeping only valid entries.
    """

    # Parse genomic coordinates from index
    coords = adata.var.index.to_series().str.extract(r'(?P<chrom>chr[\dXYM]+)[:-](?P<start>\d+)[-](?P<end>\d+)')
    #print( coords )

    # Convert start and end to integers (ignoring NaNs)
    coords['start'] = pd.to_numeric(coords['start'], errors='coerce')
    coords['end'] = pd.to_numeric(coords['end'], errors='coerce')

    # Identify valid entries (without dropping rows)
    valid_mask = coords['start'].notna() & coords['end'].notna() & (coords['start'] < coords['end'])

    # Update adata.var while keeping invalid entries as NaN
    adata.var[['chrom', 'start', 'end']] = coords

    return adata


def filter_genes_by_cre_distance(adata, distance_cutoff=100000):
    """
    Filters out genes that do not have at least one CRE within the given absolute cutoff distance.

    Args:
        - adata (AnnData): Single-cell ATAC-seq data with genes and CREs.
        - distance_cutoff (int): Maximum absolute distance for a CRE to be associated with a gene.

    Modifies:
        - Filters out genes in adata.var that do not have any CRE within the cutoff distance.
    """

    # Ensure 'cre_distances' exists in varm
    if 'cre_distances' not in adata.varm:
        raise ValueError("'cre_distances' not found in adata.varm")

    # Identify genes with at least one valid CRE distance within the cutoff
    valid_genes = [
        i for i, dists in enumerate(adata.varm['cre_distances']) 
        if isinstance(dists, (list, np.ndarray)) and 
           any(isinstance(d, (int, float)) and not np.isinf(d) and not np.isnan(d) and abs(d) <= distance_cutoff for d in dists)
    ]

    # Filter adata to keep only valid genes
    adata = adata[:, valid_genes]
 
    return adata


def get_cre_tuples(cre_adata, bin_size=1000):
    """
    Extracts CRE tuples (chromosome, start, end) with optional binning.

    Args:
        - cre_adata: AnnData object with CRE information in .var
        - bin_size: Size of the genomic binning (default: 1000)

    Returns:
        - List of (chromosome, bin_start, bin_end) tuples
    """
    cre_chroms = cre_adata.var['chrom'].to_numpy()
    cre_starts = cre_adata.var['start'].to_numpy()
    cre_ends = cre_adata.var['end'].to_numpy()

    if bin_size:
        cre_midpoints = (cre_starts + cre_ends) // 2  # Midpoint of CRE
        cre_bin_starts = (cre_midpoints // bin_size) * bin_size  # Round down to nearest bin
        cre_bin_ends = cre_bin_starts + bin_size # Round up to nearest bin
    else:
        cre_bin_starts = cre_starts
        cre_bin_ends = cre_ends      

    return list(zip(cre_chroms, cre_bin_starts, cre_bin_ends))


def map_cres_to_genes(gene_adata, cre_adata, max_cres_per_gene=10):
    """
    Map cis-regulatory elements (CREs) to genes based on genomic proximity, considering gene strand orientation.

    Args:
        - gene_adata (AnnData): AnnData object with genes and their TSS coordinates and strand information.
        - cre_adata (AnnData): AnnData object with cis-regulatory elements (CREs) and their genomic coordinates.
        - max_cres_per_gene (int): Maximum number of CREs to associate with each gene.

    Modifies:
        - Adds 'cre_names' and 'cre_distances' as 2D NumPy arrays in `gene_adata.varm`.
    """

    # Extract TSS and strand information from gene_adata
    gene_chroms = gene_adata.var['chrom'].to_numpy()
    gene_tss = gene_adata.var['tss'].to_numpy()
    gene_strands = gene_adata.var['strand'].to_numpy()  # '+' or '-'

    # Extract CRE information from cre_adata
    cre_chroms = cre_adata.var['chrom'].to_numpy()
    cre_starts = cre_adata.var['start'].to_numpy()
    cre_ends = cre_adata.var['end'].to_numpy()
    cre_midpoints = (cre_starts + cre_ends) // 2  # Use midpoint of CRE for mapping

    num_genes = len(gene_adata.var)
 
    # Initialize fixed-size 2D arrays (default to -1 for indices and NaN for distances)
    gene_cre_indices = np.full((num_genes, max_cres_per_gene), -1, dtype=int)
    gene_cre_distances = np.full((num_genes, max_cres_per_gene), np.nan, dtype=float)

    for gene_idx, (g_chrom, g_tss, g_strand) in enumerate(zip(gene_chroms, gene_tss, gene_strands)):
        if g_chrom is None or g_tss is None or g_strand is None:
            continue

        # Filter CREs on the same chromosome
        same_chrom_mask = (cre_chroms == g_chrom)
        cre_positions = cre_midpoints[same_chrom_mask]

        if len(cre_positions) == 0:
            continue

        # Compute distances (consider gene strand orientation)
        distances = cre_positions - g_tss
        if g_strand == "-":
            distances = -distances  # Flip sign for reverse strand genes

        cre_indices = np.where(same_chrom_mask)[0]  # Get original indices

        # Sort by absolute distance
        sorted_indices = np.argsort(np.abs(distances))[:max_cres_per_gene]
        selected_indices = cre_indices[sorted_indices]
        selected_distances = distances[sorted_indices]
        #print( gene_idx, g_tss, selected_distances )

        # Store in 2D arrays (padding handled by initialization)
        gene_cre_indices[gene_idx, :len(selected_indices)] = selected_indices
        gene_cre_distances[gene_idx, :len(selected_distances)] = selected_distances

    # Store results in gene_adata.varm
    # gene_adata.varm['cre_indices'] = gene_cre_indices

    valid_mask = gene_cre_indices != -1
    valid_cre_names = np.where(valid_mask, np.array(cre_adata.var_names)[gene_cre_indices], None)
    gene_adata.varm['cre_names'] = valid_cre_names

    gene_adata.varm['cre_distances'] = gene_cre_distances

    return gene_adata


def compute_rp_correlations(gex_adata):
    """
    Compute the correlation between gene expression and regulatory potential:
    - Across genes for each cell
    - Across cells for each gene
    - Summarize statistics for all genes and cells.

    Args:
        - gex_adata: AnnData object with gene expression in `X` 
          and regulatory potential in `varm["regulatory_potential"]`.

    Returns:
        - cellwise_corrs: Array of correlation coefficients for each cell
        - genewise_corrs: Array of correlation coefficients for each gene
        - stats: Dictionary summarizing mean/variance of correlations
    """
    # Extract gene expression (cells x genes)
    gene_ex = gex_adata.X.toarray() if sp.issparse(gex_adata.X) else gex_adata.X

    # Extract regulatory potential (cells x genes)
    reg_potential = gex_adata.varm["regulatory_potential"].T  # Ensure shape (cells x genes)

    num_cells, num_genes = gene_ex.shape

    # Compute correlations across genes for each cell
    cellwise_corrs = np.full(num_cells, np.nan)
    for i in range(num_cells):
        if np.std(gene_ex[i, :]) > 0 and np.std(reg_potential[i, :]) > 0:
            cellwise_corrs[i], _ = pearsonr(gene_ex[i, :], reg_potential[i, :])

    # Compute correlations across cells for each gene
    genewise_corrs = np.full(num_genes, np.nan)
    for j in range(num_genes):
        if np.std(gene_ex[:, j]) > 0 and np.std(reg_potential[:, j]) > 0:
            genewise_corrs[j], _ = pearsonr(gene_ex[:, j], reg_potential[:, j])

    # Compute statistics
    stats = {
        "cellwise_mean_corr": np.nanmean(cellwise_corrs),
        "cellwise_var_corr": np.nanvar(cellwise_corrs),
        "genewise_mean_corr": np.nanmean(genewise_corrs),
        "genewise_var_corr": np.nanvar(genewise_corrs)
    }

    return cellwise_corrs, genewise_corrs, stats


def compute_regulatory_potential(atac_adata, gex_adata, decay_factor=20000):
    """
    Compute the regulatory potential for each gene in each cell.
    
    Parameters:
    - atac_adata: AnnData object containing chromatin accessibility data (cells x CREs).
    - gex_adata: AnnData object containing gene metadata, including CRE assignments.
    - decay_factor: Controls the exponential decay rate based on CRE distance.

    Updates gex_adata.varm with regulatory potential for each gene across all cells.
    """
    
    num_cells = atac_adata.shape[0]
    num_genes = gex_adata.shape[1]
    
    # Initialize an empty matrix for regulatory potentials (genes x cells)
    reg_potential = np.zeros((num_genes, num_cells))

    for gene_idx, gene in enumerate(gex_adata.var_names):
        cre_names = gex_adata.varm['cre_names'][gene_idx]
        cre_distances = gex_adata.varm['cre_distances'][gene_idx]
        
        if len(cre_names) == 0:
            continue  # Skip genes without associated CREs

        # Retrieve chromatin accessibility for the selected CREs
        cre_accessibility = atac_adata[:, cre_names].X  # (cells x CREs)

        # Compute distance weights using an exponential decay function
        distance_weights = 2 ** (-np.abs(cre_distances) / decay_factor)

        # Compute weighted sum across CREs for each cell
        reg_potential[gene_idx, :] = np.dot(cre_accessibility, distance_weights)

    # Store per-cell regulatory potentials in varm (genes x cells)
    gex_adata.varm["regulatory_potential"] = reg_potential


def print_gene_reg_info(atac_adata, gex_adata, gene_list=['SLAMF7', 'JCHAIN', 'PRDM1']):
    """
    Print metadata, gene expression, and regulatory elements for selected genes.
    
    Parameters:
    - atac_adata: AnnData object containing chromatin accessibility data (cells x CREs).
    - gex_adata: AnnData object containing gene metadata, gene expression, and regulatory potential.
    - gene_list: List of gene symbols to display.
    """
    
    for gene in gene_list:
        if gene not in gex_adata.var["gene_symbols"].values:
            print(f"Gene {gene} not found in gex_adata.")
            continue

        # Get the index of the gene
        gene_idx = np.where(gex_adata.var["gene_symbols"].values == gene)[0]
        
        if len(gene_idx) == 0:
            print(f"Skipping {gene}: No valid index found.")
            continue
        
        gene_idx = gene_idx[0]  # Convert to integer index

        # Retrieve gene metadata
        chrom = gex_adata.var['chrom'].iloc[gene_idx]
        tss = gex_adata.var['tss'].iloc[gene_idx]
        strand = gex_adata.var['strand'].iloc[gene_idx]

        # Retrieve gene expression levels across cell states
        gene_expression = gex_adata[:, gene_idx].X

        # Retrieve CRE info
        cre_names = gex_adata.varm['cre_names'][gene_idx]
        cre_distances = gex_adata.varm['cre_distances'][gene_idx]

        # Retrieve regulatory potential for this gene across all cells
        regulatory_potential = gex_adata.varm['regulatory_potential'][gene_idx]

        # Sort CREs by distance to TSS
        sorted_indices = np.argsort(np.abs(cre_distances))
        sorted_cre_names = cre_names[sorted_indices]
        sorted_cre_distances = cre_distances[sorted_indices]

        # Retrieve chromatin accessibility for CREs
        chromatin_accessibility = atac_adata[:, sorted_cre_names].X  # (cells x CREs)

        # Print gene metadata
        print(f"\nGene: {gene}")
        print(f"  Chromosome: {chrom}, TSS: {tss}, Strand: {strand}")

        # Format and print gene expression
        formatted_gene_expression = '\t'.join(f'{g:.1f}' for g in gene_expression.ravel()[:8])
        print(f"  Gene Expression (across cell states): {formatted_gene_expression}")

        # Format and print regulatory potential
        formatted_reg_potential = '\t'.join(f'{rp:.2f}' for rp in regulatory_potential[:8])
        print(f"  Regulatory Potential (across cell states): {formatted_reg_potential}")

        # Format and print sorted CRE distances
        formatted_cre_distances = '\t'.join(f'{d:.1f}' for d in sorted_cre_distances[:8])
        print(f"  CRE Distances (sorted by proximity to TSS): {formatted_cre_distances}")

        # Format and print chromatin accessibility for first few CREs
        print(f"  Chromatin Accessibility at Closest CREs:")
        for i in range(min(4, chromatin_accessibility.shape[1])):  # Print up to 4 rows
            formatted_accessibility = '\t'.join(f'{a:.1f}' for a in chromatin_accessibility[:, i][:8])
            print(f"    {formatted_accessibility}")


def find_common_cells(adata1, adata2):
    '''
    Finds common cells between two dataframes and concatenates features
    to form the joint representation. 

    Parameters
    ----------
    adata1, adata2 : anndata.AnnData
        Two AnnData objects from which to construct joint representation.
        Order (ATAC or RNA) does not matter. 

    Returns
    -------

    adata1 : anndata.AnnData
    adata2 : anndata.AnnData
        Adata objects containing only cells common between both input adatas..
    '''
    obs_1, obs_2 = adata1.obs_names.values, adata2.obs_names.values
    shared_cells = np.intersect1d(obs_1, obs_2)

    num_shared_cells = len(shared_cells)
    if num_shared_cells == 0:
        raise ValueError('No cells/obs are shared between these two datasets. Make sure .obs_names is formatted identically between datasets.')

    total_cells = len(obs_1) + len(obs_2) - num_shared_cells
    adata1 = adata1[shared_cells].copy()
    adata2 = adata2[shared_cells].copy()
    return adata1, adata2


def filter_atac_by_cre_names(atac_adata, cre_names):
    """
    Filters atac_adata to retain only the CREs listed in cre_names.

    Args:
        atac_adata: AnnData object containing CRE information.
        cre_names: 2D array or list of lists with CRE names per gene.

    Returns:
        AnnData object with only the selected CREs.
    """
    # Flatten the list and remove None values
    selected_cres = set(cre for cre_list in cre_names if cre_list is not None for cre in cre_list)

    # Ensure we only select valid indices
    selected_cres = list(selected_cres.intersection(atac_adata.var_names))

    # Subset atac_adata based on the selected CREs
    atac_filtered = atac_adata[:, selected_cres].copy()

    return atac_filtered


def get_highly_variable_genes(adata, n_top_genes=5000):
    # Identify the top 5000 highly variable genes using Seurat v3 method
    #sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Get the list of top 5000 highly variable genes
    top_genes = adata.var[adata.var['highly_variable']].index.tolist()

    # Print the first few genes as a preview
    print(top_genes[:10])  # Show the first 10 genes

    # Optionally, filter the AnnData object to keep only highly variable genes
    adata_filtered = adata[:, top_genes]
    return adata_filtered


def print_adata_summary(adata):
    # Check the main components
    print("\n--- Observations (obs) ---")
    print(adata.obs.head())  # Metadata for cells
    
    print("\n--- Variables (var) ---")
    print(adata.var.head())  # Metadata for features (e.g., genes or peaks)
    
    # Check varm, obsm, uns, layers, etc.
    print("\n--- varm (Variable metadata matrices) ---")
    print(adata.varm.keys())
    
    print("\n--- obsm (Observation metadata matrices) ---")
    print(adata.obsm.keys())
    
    print("\n--- uns (Unstructured data) ---")
    print(adata.uns.keys())
    
    print("\n--- Layers ---")
    print(adata.layers.keys())

    print("\n--- Matrix (X) ---")
    if adata.X is not None:
        print(type(adata.X), adata.X.shape)  # Expression or accessibility matrix
        print(adata.X)

 
def summarize_highly_expressed_genes(adata):
    """
    Finds the cell where each gene is most highly expressed and summarizes the top-expressed genes for each cell.
    
    Parameters:
    - adata: AnnData object (cells x genes)

    Returns:
    - summary_df: DataFrame summarizing which genes are most highly expressed in each cell
    """
    # Convert sparse matrix to dense if needed
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X

    # Get gene and cell names
    gene_names = adata.var_names
    cell_names = adata.obs_names if adata.obs_names is not None else np.arange(X.shape[0])

    # Find the cell index where each gene is maximally expressed
    max_cell_idx = np.argmax(X, axis=0)

    # Create a dictionary mapping each cell to its most highly expressed genes
    cell_to_genes = {}
    for gene, cell_idx in zip(gene_names, max_cell_idx):
        cell = cell_names[cell_idx]

        # Print gene, cell_idx, and expression values across all cells
        # print(f"Gene: {gene}, Max Cell Index: {cell_idx}, Expression Across Cells: {X[:, gene_names.get_loc(gene)]}")

        if cell not in cell_to_genes:
            cell_to_genes[cell] = []
        cell_to_genes[cell].append(gene)

    # Convert to a DataFrame for better readability
    summary_df = pd.DataFrame([(cell, genes) for cell, genes in cell_to_genes.items()], columns=["Cell", "Top Genes"])

    return summary_df


def get_embeddings(embed_file='', regions=[]):
    """
    Retrieve embeddings for a given set of genomic regions from an HDF5 file.

    Args:
        - embed_file (str): Path to the HDF5 file containing embeddings.
        - regions (list of tuples): List of regions in the format (chrom, start, end).

    Returns:
        - embeddings (numpy.ndarray): Array of shape (n_regions, dim_embeddings).
    """
    # Create region lookup index
    region_lookup = embedding_utils.make_hdf5_region_index(embed_file)

    # Load the embeddings mapping from the HDF5 file
    embedding_lookup = embedding_utils.make_region_to_embeddings(embed_file)
    
    # Get embedding dimension
    _, dim = embedding_utils.get_embedding_dim(embed_file)

    # Get the default embedding for missing regions
    null_embedding = embedding_utils.get_out_of_vocab_embeddings(embed_file)

    # Initialize an array to store embeddings
    embeddings = np.zeros((len(regions), dim))

    # Retrieve embeddings for each region
    for i, region in enumerate(regions):
        embeddings[i] = embedding_lookup.get(region, null_embedding)  # Use null_embedding if region is missing

    # Count the number of non-null embeddings
    num_non_null = sum(1 for emb in embeddings if not np.array_equal(emb, null_embedding))
    print(f"Number of regions with non-null embeddings: {num_non_null}")

    return embeddings


class GeneRegData:

    def __init__(self, gene_adata=None, atac_adata=None, max_cres_per_gene=10):
        self.gene_adata = gene_adata
        self.atac_adata = atac_adata
        self.max_cres_per_gene = max_cres_per_gene
        print('Loading data...')
        self.load_data()
        print('Initializing matrices...')
        self.train_val_test_split( train_frac=0.7, val_frac=0.30, test_frac=0.00, seed=42)
        print('Initializing matrices...')
        self.initialize_matrices()
        #print('Populating matrices...')
        #self.populate_cre_matrices(gene_set_name='train')
        #self.populate_gene_subset_matrices(gene_set_name='train')


    def load_data(self):
        """Extracts gene expression, CRE accessibility, embeddings, and nearest CRE distances from AnnData objects."""
        #self.gene_cre_names = self.gene_adata.varm['cre_names'][:, :self.max_cres_per_gene]
        self.gene_cre_names_all = self.gene_adata.varm['cre_names'][:, :self.max_cres_per_gene]
        #self.gene_cre_distances = self.gene_adata.varm['cre_distances'][:, :self.max_cres_per_gene]
        self.gene_cre_distances_all = self.gene_adata.varm['cre_distances'][:, :self.max_cres_per_gene]

        # Convert gene expression matrix to PyTorch tensor
        # self.gene_expression_tensor = torch.tensor(self.gene_adata.X.toarray(), dtype=torch.float32)
        # Check if X is sparse before converting
        if sp.issparse(self.gene_adata.X):
            #self.gene_expression_tensor = torch.tensor(self.gene_adata.X.toarray(), dtype=torch.float32)
            self.gene_expression_all = torch.tensor(self.gene_adata.X.toarray(), dtype=torch.float32)
        else:
            #self.gene_expression_tensor = torch.tensor(self.gene_adata.X, dtype=torch.float32)
            self.gene_expression_all = torch.tensor(self.gene_adata.X, dtype=torch.float32)

        # Convert accessibility matrix to sparse PyTorch tensor
        if sp.issparse(self.atac_adata.X):
            #self.cre_accessibility_tensor = self.anndata_to_torch_sparse(self.atac_adata.X)
            self.cre_accessibility_all = self.anndata_to_torch_sparse(self.atac_adata.X)
        else:
            #self.cre_accessibility_tensor = torch.tensor(self.atac_adata.X, dtype=torch.float32)
            self.cre_accessibility_all = torch.tensor(self.atac_adata.X, dtype=torch.float32)

        # Convert embeddings to PyTorch tensor
        # TODO add an option to exclude embeddings found in AnnData object
        #self._cre_embeddings_tensor = torch.tensor(self.atac_adata.varm['embeddings'], dtype=torch.float32)
        # NOTE: embeddings from the AnnData file are possibly combined in before loading
        ## TODO check self._cre_embeddings_all = torch.tensor(self.atac_adata.varm['embeddings'], dtype=torch.float32)
        # these are embeddings that are not aligned with genes

        if 'embeddings' in self.atac_adata.varm.keys():
            self.cre_embeddings_ref = torch.tensor(self.atac_adata.varm['embeddings'], dtype=torch.float32)
        else:
            self.cre_embeddings_ref = None

        # Create a mapping of CRE names to indices
        self.cre_name_to_index_all = {name: i for i, name in enumerate(self.atac_adata.var_names)}
        print('___ data loaded from adata ___' )
        print( 'gene_expression_all:', self.gene_expression_all.shape )
        print( 'cre_accessibility_all:', self.cre_accessibility_all.shape )
        print('______________________________' )


    def train_val_test_split(self, train_frac=0.7, val_frac=0.30, test_frac=0.00, seed=42): 
        """
        Splits gene indices into training, validation, and test sets.
 
        Parameters:
        - train_frac (float): Fraction of genes for training (default: 70%).
        - val_frac (float): Fraction of genes for validation (default: 15%).
        - test_frac (float): Fraction of genes for testing (default: 15%).
        - seed (int): Random seed for reproducibility.
 
        Returns:
        - updates: train_genes, val_genes and test_genes
        """

        assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1"
        _, n_genes = self.gene_expression_all.shape 
 
        # Set random seed for reproducibility
        np.random.seed(seed)
 
        # Generate shuffled indices
        indices = np.random.permutation(n_genes)
 
        # Compute split sizes
        train_size = int(train_frac * n_genes)
        val_size = int(val_frac * n_genes)
 
        # Split indices
        self.train_genes = indices[:train_size].tolist()
        self.val_genes = indices[train_size:train_size + val_size].tolist()
        self.test_genes = indices[train_size + val_size:].tolist()


    @staticmethod
    def anndata_to_torch_sparse(adata_matrix):
        """Converts an AnnData sparse matrix to a PyTorch sparse COO tensor."""
        if not sp.issparse(adata_matrix):
            raise ValueError("Input matrix must be a sparse matrix.")
        coo = adata_matrix.tocoo() if not isinstance(adata_matrix, sp.coo_matrix) else adata_matrix
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32)

    @staticmethod
    def assign_cre_class(dist):
        """Assigns CRE class based on proximity to the gene."""
        PROXIMAL_RANGE = 1000  # Example threshold
        return torch.where(
            (dist.abs() <= PROXIMAL_RANGE), PROXIMAL_INDEX, 
            torch.where(dist < 0, UPSTREAM_INDEX, DOWNSTREAM_INDEX)
        )

    def initialize_matrices(self):
        """Initialize distance, CRE, and embedding matrices."""
        #num_cells, num_genes = self.gene_expression_tensor.shape
        num_cells, num_genes = self.gene_expression_all.shape
        #self.cre_tensor = torch.zeros(num_cells, num_genes, self.max_cres_per_gene)
        self.cre_all = torch.zeros(num_cells, num_genes, self.max_cres_per_gene)
        #self.distances_tensor = torch.full((num_genes, self.max_cres_per_gene), float("inf"))
        self.distances_all = torch.full((num_genes, self.max_cres_per_gene), float("inf"))
        #self.cre_class_tensor = torch.full((num_genes, self.max_cres_per_gene), -1, dtype=torch.int)
        self.cre_class_all = torch.full((num_genes, self.max_cres_per_gene), -1, dtype=torch.int)
        #self.cre_indices_tensor = torch.full((num_genes, self.max_cres_per_gene), -1, dtype=torch.int)
        self.cre_indices_all = torch.full((num_genes, self.max_cres_per_gene), -1, dtype=torch.int)
        ## TODO check embedding_dim = self._cre_embeddings_all.shape[1]
        if self.cre_embeddings_ref is None:
            embedding_dim = 0
        else:
            embedding_dim = self.cre_embeddings_ref.shape[1]
        #self.cre_embeddings_tensor = torch.zeros((embedding_dim, num_genes, self.max_cres_per_gene))
        # These are cre_embeddings aligned with genes
        # TODO check not needed self.cre_embeddings_all = torch.zeros((embedding_dim, num_genes, self.max_cres_per_gene))


    def populate_gene_subset_matrices(self, gene_set_name=None, debug=True):
        """Populate CRE activity, distance, and embedding matrices for a subset of genes.
        args: 
            - gene_set_name (str): options are train, validate, test, all  
        side-effects:
            - modifies num_genes and tensors used in regression model:
                  self.cre_tensor
                  self.distances_tensor
                  self.cre_indices_tensor
                  self.cre_class_tensor
                  self.cre_embeddings_tensor
                  self.num_genes # NOTE: num_genes is the number in the gene set
        """

        if gene_set_name == 'train':
            self.gene_indices = self.train_genes
        elif gene_set_name == 'validate':
            self.gene_indices = self.val_genes
        elif gene_set_name == 'test':
            self.gene_indices = self.test_genes
        elif gene_set_name is 'all':
            self.gene_indices = list(range(self.gene_cre_names_all.shape[0]))
        else:
            raise ValueError(f"Unknown gene_set_name: {gene_set_name}\
                valid choices are train, validate, test and all")

        self.num_genes = len(self.gene_indices)
        assert ( self.num_genes > 0 ), "The number of genes in the set much be positive."

        dense_cre_accessibility = self.cre_accessibility_all.to_dense()

        # Normalize by scaling
        sums = dense_cre_accessibility.sum(dim=1, keepdim=True)  # shape: [num_cell_types, 1]
        sums = torch.clamp(sums, min=1e-8)
        normalized_cre_accessibility = 1.0e6 * dense_cre_accessibility / sums
 
        if debug:
            print('populate gene set name:', gene_set_name)
            print('total number of genes:', self.gene_cre_names_all.shape[0])
            print('subset gene count:', len(self.gene_indices))
            print('sums shape:', sums.shape)
            print(f'gene_set_name: {gene_set_name} indices sample:', self.gene_indices[0:10])
 
        # Allocate subset tensors for the selected gene set
        num_cell_types = self.cre_all.shape[0]
        max_cres = self.cre_all.shape[2]

        if self.cre_embeddings_ref is None:
            embedding_dim = 0
        else:
            embedding_dim = self.cre_embeddings_ref.shape[1]
  
        self.gene_expression_tensor = self.gene_expression_all[:, self.gene_indices]

        self.cre_tensor = torch.zeros((num_cell_types, self.num_genes, max_cres),\
            dtype=self.cre_all.dtype, device=self.cre_all.device)
        self.distances_tensor = torch.zeros((self.num_genes, max_cres),\
            dtype=self.distances_all.dtype, device=self.distances_all.device)
        self.cre_indices_tensor = torch.zeros((self.num_genes, max_cres),\
            dtype=self.cre_indices_all.dtype, device=self.cre_indices_all.device)
        self.cre_class_tensor = torch.zeros((self.num_genes, max_cres),\
            dtype=self.cre_class_all.dtype, device=self.cre_class_all.device)

        if embedding_dim > 0:
            self.cre_embeddings_tensor = torch.zeros((embedding_dim, self.num_genes, max_cres),\
                dtype=self.cre_embeddings_ref.dtype, device=self.cre_embeddings_ref.device)

        for i, gene_idx in enumerate(self.gene_indices):  # i is the index in the subset
            cre_names = self.gene_cre_names_all[gene_idx]
     
            valid_mask = torch.tensor(cre_names != None, dtype=torch.bool, device=self.cre_indices_tensor.device)
            cre_names = cre_names[valid_mask]
     
            if len(cre_names) == 0:
                continue
     
            cre_indices = [self.cre_name_to_index_all[name] for name in cre_names if name in self.cre_name_to_index_all]
            if len(cre_indices) == 0:
                continue
     
            cre_indices_tensor = torch.tensor(cre_indices,\
                dtype=self.cre_indices_all.dtype, device=self.cre_indices_all.device)
            valid_indices = torch.where(valid_mask)[0]
     
            # use _all to mark tensor with all genes
            self.cre_tensor[:, i, valid_indices] = normalized_cre_accessibility[:, cre_indices]
            self.distances_tensor[i, valid_indices] = torch.tensor(
                self.gene_cre_distances_all[gene_idx, valid_indices],\
                    dtype=torch.float32, device=self.distances_tensor.device
            )
            self.cre_indices_tensor[i, valid_indices] = cre_indices_tensor
            self.cre_class_tensor[i, valid_indices] = self.assign_cre_class(self.distances_tensor[i, valid_indices]).to(torch.int)
            ## TODO check self.cre_embeddings_tensor[:, i, valid_indices] = self._cre_embeddings_all[cre_indices].T
            # NOTE: we load embeddings from ref using cre_indices
            if embedding_dim > 0:
                self.cre_embeddings_tensor[:, i, valid_indices] = self.cre_embeddings_ref[cre_indices].T

     
    def summarize_cre_tensors(self):
        def summarize(tensor, name):
            if tensor.is_sparse:
                tensor = tensor.to_dense()
            print(f"{name}: shape={tuple(tensor.shape)}, mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
            print("====== CRE Matrix Summary ======")
        summarize(self.cre_tensor, "cre_tensor")
        summarize(self.distances_tensor, "distances_tensor")
        summarize(self.cre_indices_tensor, "cre_indices_tensor")
        summarize(self.cre_class_tensor, "cre_class_tensor")
        summarize(self.cre_embeddings_tensor, "cre_embeddings_tensor")
        print("================================")


    def get_processed_data(self):
        """Return processed tensors."""
        return (
            self.gene_expression_tensor,
            self.cre_tensor,
            self.distances_tensor,
            self.cre_class_tensor,
            self.cre_embeddings_tensor
        )


def combine_embeddings( adata, embed_file, method="concatenate"):
    """
    Aligns and combines internal and external embeddings for ATAC-seq data.

    Parameters:
    - adata: AnnData object containing internal embeddings in varm["embeddings"]
    - embed_file: Path to the external embeddings file
    - method: How to combine embeddings ("concatenate")

    Returns:
    - Updates atac_adata.varm["embeddings"] with the combined embeddings
    """
    
    # Load external embeddings based on genomic regions
    regions = get_cre_tuples(adata, bin_size=1000)
    if embed_file:
        external_embeddings = get_embeddings(embed_file=embed_file, regions=regions)
        print("shape external embeddings:", external_embeddings.shape)
    else:
        external_embeddings = None

    if "embeddings" in adata.varm.keys():
        internal_embeddings = adata.varm["embeddings"]
        print("shape internal embeddings:", internal_embeddings.shape)
    else:
        internal_embeddings = None
        print("no internal embeddings")

    if internal_embeddings is not None and external_embeddings is None:
        combined_embeddings = internal_embeddings
    elif external_embeddings is not None and internal_embeddings is None:
        combined_embeddings = external_embeddings
    elif (external_embeddings is not None) and (internal_embeddings is not None):
        # Ensure row alignment
        if ( internal_embeddings.shape[0] != external_embeddings.shape[0] ):
            raise ValueError("Mismatch in number of regions between internal and external embeddings.")
        # Concatenate embeddings along feature axis
        if method == "concatenate":
            combined_embeddings = np.concatenate([internal_embeddings, external_embeddings], axis=1)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from 'concatenate'.")
    else:
        combined_embeddings = None 

    # Store in adata
    if combined_embeddings is not None:
        adata.varm["embeddings"] = combined_embeddings
        print("shape combined embeddings:", combined_embeddings.shape)

    return adata


def read_atac_data(atac_file):
    atac_adata = sc.read_h5ad(atac_file)
    return parse_cre_genomic_coords(atac_adata)


def read_gex_data(gex_file):
    if gex_file.endswith(".h5ad"):
        return sc.read_h5ad(gex_file)
    elif gex_file.endswith(".h5"):
        return sc.read_10x_h5(gex_file)
    else:
        raise ValueError(f"Unsupported file format: {gex_file}")


def filter_gex_data(gex_adata, n_genes=2000, min_counts=50):
    """
    Filter for highly variable genes with enough counts
    """
    sc.pp.filter_genes(gex_adata, min_counts=min_counts)
    gex_adata.layers["raw_counts"] = gex_adata.X.copy()
    sc.pp.normalize_total(gex_adata, target_sum=1e4)
    sc.pp.log1p(gex_adata)
    gex_adata = get_highly_variable_genes(gex_adata, n_top_genes=n_genes)
    gex_adata.X = gex_adata.layers["raw_counts"].copy()
    return gex_adata


def normalize_cell_names(adata):
    # NOTE: check for samples with multiple batches! 
    adata.obs_names = adata.obs_names.str.split("-").str[0]
    return adata


# TODO def annotate_and_map_genes(gex_adata, atac_adata, max_cres_per_gene):
def annotate_and_map_genes(gex_adata, atac_adata, max_cres_per_gene, 
    gene_alias_file=ALIAS_FILE, 
    gene_ref_file=TSS_FILE ):

    gex_adata = annotate_gene_symbols(gex_adata)
    gex_adata = annotate_gene_tss( 
        gex_adata, 
        gene_alias_file=gene_alias_file, 
        gene_ref_file=gene_ref_file 
    )
    gex_adata = map_cres_to_genes(gex_adata, atac_adata, max_cres_per_gene=max_cres_per_gene)
    return filter_genes_by_cre_distance(gex_adata)


def compute_and_visualize_regulatory_potential(atac_adata, gex_adata):
    compute_regulatory_potential(atac_adata, gex_adata, decay_factor=10000)
    cellwise_corrs, genewise_corrs, stats = compute_rp_correlations(gex_adata)
    print("Correlation coefficients per cell:", cellwise_corrs)
    print(stats)
    plot_reg_potential_vs_expression_per_cell(gex_adata, max_cells=5)


def model_setup(atac_file='', gex_file='', 
    embed_file='', 
    gene_alias_file='', 
    gene_ref_file='',
    max_cres_per_gene=30, num_genes=2000, use_local_embeddings=True):

    atac_adata = read_atac_data(atac_file)
    gex_adata = read_gex_data(gex_file)

    print_adata_summary(atac_adata)
    print_adata_summary(gex_adata)

    atac_adata = normalize_cell_names(atac_adata)
    gex_adata = normalize_cell_names(gex_adata)
    atac_adata, gex_adata = find_common_cells( atac_adata, gex_adata )
    assert( atac_adata and gex_adata )

    gex_adata = filter_gex_data(gex_adata, n_genes=num_genes)
    summary_df = summarize_highly_expressed_genes(gex_adata)
    print(summary_df)

    # TODO gex_adata = annotate_and_map_genes(gex_adata, atac_adata, max_cres_per_gene)
    gex_adata = annotate_and_map_genes(
        gex_adata, atac_adata, max_cres_per_gene,
        gene_alias_file = gene_alias_file, 
        gene_ref_file = gene_ref_file
    )

    assert( gex_adata )
    
    sc.pp.normalize_total(atac_adata, target_sum=1000000.0)
    # TODO compute_and_visualize_regulatory_potential(atac_adata, gex_adata)

    atac_adata = filter_atac_by_cre_names(atac_adata, gex_adata.varm['cre_names'])
    assert( atac_adata )

    print( 'use_local_embeddings:', use_local_embeddings )
    if use_local_embeddings == False and "embeddings" in atac_adata.varm.keys():
        del atac_adata.varm["embeddings"]

    atac_adata = combine_embeddings(atac_adata, embed_file, method="concatenate")
    assert( atac_adata )

    return GeneRegData(gene_adata=gex_adata, atac_adata=atac_adata, max_cres_per_gene=max_cres_per_gene)


if __name__ == '__main__':

    #atac_file = "/Users/len/Projects/MAESTRO/MAESTRO-master/test/HodgkinLymphoma/pilot/Analysis/quick_count_peak_count_ppmi_lsi.h5ad"
    #gex_file = "/Users/len/Projects/MAESTRO/MAESTRO-master/test/HodgkinLymphoma/pilot/raw_feature_bc_matrix.h5"
    #atac_file = "adata_atac.h5ad"
    #atac_file = "adata_atac_aggregated.h5ad"
    #gex_file = "adata_gex.h5ad"
    #gex_file = "adata_gex_aggregated.h5ad"

    atac_file = "adata_atac_aggregated_eryth_mono_ppmi.h5ad"
    gex_file = "adata_gex_aggregated_eryth_mono.h5ad"

    embed_file = EMBED_FILE
    model_setup(
        atac_file=atac_file, 
        gex_file=gex_file, 
        embed_file=embed_file,
        gene_alias_file=ALIAS_FILE,
        gene_ref_file=TSS_FILE
    )

